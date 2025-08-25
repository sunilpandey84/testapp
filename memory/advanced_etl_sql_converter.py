#!/usr/bin/env python3
"""
Advanced ETL to SQL Converter Agent
A comprehensive solution for converting various ETL tool logic to ANSI SQL with emphasis on Spark SQL
Built for data stewards to easily translate ETL processes into standardized SQL
"""

import json
import asyncio
import logging
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import ast
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from dotenv import load_dotenv
load_dotenv()
# For LLM integration
try:
    import openai  # pip install openai
    from anthropic import Anthropic  # pip install anthropic
    from langchain_google_genai import ChatGoogleGenerativeAI  # pip install langchain-google-genai
except ImportError:
    print("Please install required packages: pip install openai anthropic")

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ETLToolType(Enum):
    """Supported ETL tools for conversion"""
    INFORMATICA = "informatica"
    PYTHON_PANDAS = "python_pandas"
    PYSPARK = "pyspark"
    TALEND = "talend"
    PENTAHO = "pentaho"
    SSIS = "ssis"
    DATASTAGE = "datastage"
    STORED_PROCEDURE = "stored_procedure"
    SQL_SCRIPT = "sql_script"
    CUSTOM = "custom"

class DatabaseType(Enum):
    """Supported target database types"""
    SPARK_SQL = "spark_sql"
    POSTGRESQL = "postgresql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    MYSQL = "mysql"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    HIVE = "hive"
    GENERIC_SQL = "ansi_sql"

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 4000
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, provider_name: str = None):
        """Create LLM config from environment variables"""
        import os
        
        # Default to OpenAI if not specified
        if not provider_name:
            provider_name = os.environ.get("ETL_LLM_PROVIDER", "openai_gpt4")
        
        provider = LLMProvider(provider_name)
        
        # Get API key based on provider
        if provider == LLMProvider.OPENAI_GPT4:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")
        elif provider == LLMProvider.ANTHROPIC_CLAUDE:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            model = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        elif provider == LLMProvider.GEMINI:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        elif provider == LLMProvider.AZURE_OPENAI:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4")
            additional_params = {
                "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            }
            return cls(provider=provider, api_key=api_key, model_name=model, additional_params=additional_params)
        
        return cls(provider=provider, api_key=api_key, model_name=model)

@dataclass
class ConversionContext:
    """Comprehensive context for ETL-to-SQL conversion"""
    etl_tool: ETLToolType
    source_code: str
    target_database: DatabaseType
    schema_info: Dict[str, Any] = field(default_factory=dict)
    business_rules: List[str] = field(default_factory=list)
    data_lineage: Dict[str, Any] = field(default_factory=dict)
    table_samples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    data_quality_rules: List[Dict[str, Any]] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "etl_tool": self.etl_tool.value,
            "target_database": self.target_database.value,
            "schema_info": self.schema_info,
            "business_rules": self.business_rules,
            "data_lineage": self.data_lineage,
            "table_samples": self.table_samples,
            "data_quality_rules": self.data_quality_rules,
            "execution_context": self.execution_context
        }

@dataclass
class ConversionResult:
    """Result of an ETL-to-SQL conversion"""
    success: bool
    sql_code: Optional[str] = None
    documentation: Optional[str] = None
    lineage_diagram: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "sql_code": self.sql_code,
            "documentation": self.documentation,
            "lineage_diagram": self.lineage_diagram,
            "error": self.error,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "quality_score": self.quality_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_exception(cls, exception: Exception) -> 'ConversionResult':
        """Create result object from exception"""
        return cls(
            success=False,
            error=f"{type(exception).__name__}: {str(exception)}",
            warnings=["Conversion failed due to an error."],
            execution_time=0.0
        )

class LLMOrchestrator:
    """Orchestrates LLM calls for ETL conversion with reliability features"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self._retry_count = 3
        self._max_chunk_tokens = 3000
        
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider == LLMProvider.OPENAI_GPT4:
            return openai.OpenAI(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return Anthropic(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini integration not implemented; raise error or implement client initialization here
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            return openai.AzureOpenAI(
                api_key=self.config.api_key,
                api_version="2023-12-01-preview",
                azure_endpoint=self.config.additional_params.get('azure_endpoint')
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def convert_etl_to_sql(self, context: ConversionContext) -> ConversionResult:
        """Main LLM-powered conversion function with enhanced error handling"""
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze ETL code structure
            analysis_prompt = self._build_analysis_prompt(context)
            analysis_result = await self._call_llm_with_retry(analysis_prompt, "analysis")
            
            # Step 2: Extract data lineage
            lineage_prompt = self._build_lineage_prompt(context, analysis_result)
            lineage_result = await self._call_llm_with_retry(lineage_prompt, "lineage_extraction")
            
            # Step 3: Generate initial SQL with business context
            sql_prompt = self._build_sql_generation_prompt(context, analysis_result, lineage_result)
            sql_result = await self._call_llm_with_retry(sql_prompt, "sql_generation")
            
            # Step 4: Optimize for target database (with emphasis on Spark SQL if selected)
            optimization_prompt = self._build_optimization_prompt(sql_result, context.target_database.value)
            optimized_sql = await self._call_llm_with_retry(optimization_prompt, "optimization")
            
            # Step 5: Generate comprehensive documentation
            doc_prompt = self._build_documentation_prompt(context, optimized_sql, analysis_result, lineage_result)
            documentation = await self._call_llm_with_retry(doc_prompt, "documentation")
            
            # Step 6: Generate data quality checks for the SQL
            dq_prompt = self._build_data_quality_prompt(context, optimized_sql)
            data_quality_sql = await self._call_llm_with_retry(dq_prompt, "data_quality")
            
            # Combine the optimized SQL with data quality checks
            final_sql = self._combine_sql_with_quality_checks(optimized_sql, data_quality_sql)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Try to parse lineage_result as JSON if it's a string
            lineage_data = {}
            if isinstance(lineage_result, str):
                try:
                    # Try to extract JSON from the string
                    json_match = re.search(r'```json\s*(.*?)\s*```', lineage_result, re.DOTALL)
                    if json_match:
                        lineage_data = json.loads(json_match.group(1))
                    else:
                        # Try parsing the whole string as JSON
                        lineage_data = json.loads(lineage_result)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, leave as empty dict
                    lineage_data = {}
            
            return ConversionResult(
                success=True,
                sql_code=final_sql,
                documentation=documentation,
                lineage_diagram=lineage_data.get('lineage_diagram', ''),
                execution_time=execution_time,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'llm_provider': self.config.provider.value,
                    'model_used': self.config.model_name,
                    'target_database': context.target_database.value,
                    'analysis_summary': self._extract_analysis_summary(analysis_result),
                }
            )
            
        except Exception as e:
            logger.error(f"LLM conversion failed: {str(e)}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ConversionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                warnings=["An error occurred during the LLM-powered conversion process."]
            )

    def _combine_sql_with_quality_checks(self, main_sql: str, quality_sql: str) -> str:
        """Combine optimized SQL with data quality checks"""
        combined = f"""
-- Main ETL SQL Logic
{main_sql.strip()}

-- Data Quality Checks
{quality_sql.strip()}
"""
        return combined

    def _extract_analysis_summary(self, analysis_result: str) -> Dict[str, Any]:
        """Extract summary from analysis result"""
        # Try to extract JSON from the analysis result
        try:
            # Look for JSON pattern in the result
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON pattern, try to parse the entire string as JSON
            return json.loads(analysis_result)
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, return a simplified summary
            return {
                "summary": "Analysis completed but structured data extraction failed",
                "raw_length": len(analysis_result)
            }

    def _build_analysis_prompt(self, context: ConversionContext) -> str:
        """Build prompt for ETL code analysis with enhanced database guidance"""
        db_specific_guidance = self._get_database_specific_guidance(context.target_database)
        
        # Special handling for stored procedures
        if context.etl_tool == ETLToolType.STORED_PROCEDURE:
            return self._build_stored_procedure_analysis_prompt(context, db_specific_guidance)
        elif context.etl_tool == ETLToolType.SQL_SCRIPT:
            return self._build_sql_script_analysis_prompt(context, db_specific_guidance)
        
        # Standard ETL tool prompt
        return f"""
You are an expert data engineer analyzing ETL code for conversion to SQL.

**Task**: Analyze the following {context.etl_tool.value} code and extract:
1. Data sources and their schemas
2. Transformation logic and business rules
3. Data quality checks and validations
4. Output targets and data flow
5. Complex logic that needs special SQL handling
6. Identify any performance bottlenecks in the original code

**ETL Tool**: {context.etl_tool.value}
**Target Database**: {context.target_database.value}

{db_specific_guidance}

**Schema Context**:
{json.dumps(context.schema_info, indent=2) if context.schema_info else "No schema info provided"}

**Business Rules**:
{chr(10).join(context.business_rules) if context.business_rules else "No specific business rules provided"}

**ETL Code to Analyze**:
```
{context.source_code}
```

**Required Output Format**:
Return a JSON object with this structure:
{{
    "data_sources": [
        {{
            "name": "table_name",
            "type": "source_type",
            "columns": ["col1", "col2"],
            "key_columns": ["primary_key"]
        }}
    ],
    "transformations": [
        {{
            "type": "transformation_type",
            "description": "what this transformation does",
            "complexity": "simple|medium|complex",
            "sql_approach": "suggested SQL approach"
        }}
    ],
    "business_rules": [
        {{
            "rule": "business rule description",
            "implementation": "how to implement in SQL"
        }}
    ],
    "data_quality": [
        {{
            "check": "quality check description",
            "sql_implementation": "SQL for the check"
        }}
    ],
    "data_flow": "description of overall data flow",
    "complexity_assessment": "overall complexity rating and considerations"
}}

Analyze thoroughly and be specific about SQL implementation strategies for {context.target_database.value}.
"""

    def _build_stored_procedure_analysis_prompt(self, context: ConversionContext, db_specific_guidance: str) -> str:
        """Build specialized prompt for analyzing stored procedures"""
        return f"""
You are an expert database engineer analyzing stored procedure code for conversion to standardized SQL.

**Task**: Analyze the following stored procedure code and extract:
1. Data sources and their schemas
2. Transformation logic and procedural flow
3. Data quality checks and validations
4. Output targets and data flow
5. Complex logic that needs special SQL handling
6. Identify any performance bottlenecks or cursor operations

**Source Database Type**: Identify from the code (SQL Server, Oracle PL/SQL, PostgreSQL PL/pgSQL, MySQL, etc.)
**Target Database**: {context.target_database.value}

{db_specific_guidance}

**Schema Context**:
{json.dumps(context.schema_info, indent=2) if context.schema_info else "No schema info provided"}

**Business Rules**:
{chr(10).join(context.business_rules) if context.business_rules else "No specific business rules provided"}

**Stored Procedure Code to Analyze**:
```sql
{context.source_code}
```

**Required Output Format**:
Return a JSON object with this structure:
{{
    "procedure_info": {{
        "name": "procedure_name",
        "parameters": [
            {{ "name": "param1", "type": "data_type", "mode": "IN/OUT/INOUT" }}
        ],
        "source_dialect": "identified_dialect"
    }},
    "data_sources": [
        {{
            "name": "table_name",
            "type": "source_type",
            "schema": "schema_name"
        }}
    ],
    "data_targets": [
        {{
            "name": "table_name",
            "type": "target_type",
            "schema": "schema_name"
        }}
    ],
    "procedural_elements": [
        {{
            "type": "cursor|variable|transaction|exception",
            "description": "what this element does",
            "conversion_approach": "how to handle in standard SQL"
        }}
    ],
    "transformations": [
        {{
            "type": "transformation_type",
            "description": "what this transformation does",
            "complexity": "simple|medium|complex",
            "sql_approach": "suggested SQL approach"
        }}
    ],
    "business_rules": [
        {{
            "rule": "business rule description",
            "implementation": "how to implement in SQL"
        }}
    ],
    "data_flow": "description of overall data flow",
    "complexity_assessment": "overall complexity rating and considerations"
}}

Analyze thoroughly and be specific about SQL implementation strategies for {context.target_database.value}.
Focus on how to convert procedural elements like cursors, exception handling, and transactions into set-based operations in {context.target_database.value}.
"""

    def _build_sql_script_analysis_prompt(self, context: ConversionContext, db_specific_guidance: str) -> str:
        """Build specialized prompt for analyzing SQL scripts"""
        return f"""
You are an expert database engineer analyzing SQL script code for conversion to standardized SQL.

**Task**: Analyze the following SQL script code and extract:
1. Data sources and their schemas
2. Transformation logic and script flow
3. Data quality checks and validations
4. Output targets and data flow
5. Temporary tables and their usage
6. Identify any performance bottlenecks

**Source SQL Dialect**: Identify from the code (T-SQL, PL/SQL, PostgreSQL, MySQL, etc.)
**Target Database**: {context.target_database.value}

{db_specific_guidance}

**Schema Context**:
{json.dumps(context.schema_info, indent=2) if context.schema_info else "No schema info provided"}

**Business Rules**:
{chr(10).join(context.business_rules) if context.business_rules else "No specific business rules provided"}

**SQL Script Code to Analyze**:
```sql
{context.source_code}
```

**Required Output Format**:
Return a JSON object with this structure:
{{
    "script_info": {{
        "dialect": "identified_dialect",
        "statement_count": 10,
        "script_purpose": "brief description"
    }},
    "data_sources": [
        {{
            "name": "table_name",
            "type": "source_type",
            "schema": "schema_name"
        }}
    ],
    "data_targets": [
        {{
            "name": "table_name",
            "type": "target_type",
            "schema": "schema_name"
        }}
    ],
    "temp_tables": [
        {{
            "name": "temp_table_name",
            "purpose": "what this temp table stores",
            "conversion_approach": "how to handle in target SQL"
        }}
    ],
    "transformations": [
        {{
            "type": "transformation_type",
            "description": "what this transformation does",
            "complexity": "simple|medium|complex",
            "sql_approach": "suggested SQL approach"
        }}
    ],
    "workflow": {{
        "stages": [
            {{ "operation": "operation_type", "purpose": "stage purpose" }}
        ],
        "execution_flow": "description of execution flow"
    }},
    "business_rules": [
        {{
            "rule": "business rule description",
            "implementation": "how to implement in SQL"
        }}
    ],
    "data_flow": "description of overall data flow",
    "complexity_assessment": "overall complexity rating and considerations"
}}

Analyze thoroughly and be specific about SQL implementation strategies for {context.target_database.value}.
Focus on converting any dialect-specific syntax and optimizing for {context.target_database.value}.
"""

    def _build_lineage_prompt(self, context: ConversionContext, analysis: str) -> str:
        """Build prompt for extracting data lineage"""
        return f"""
You are an expert data lineage analyst mapping data flows from ETL code.

**Task**: Extract detailed data lineage information from the analyzed ETL code.

**Analysis Results**:
{analysis}

**Original ETL Code**:
```
{context.source_code}
```

**Requirements**:
1. Map source tables/fields to target tables/fields
2. Identify transformations applied to each field
3. Document join conditions and filters
4. Note any derived fields and their calculations

**Output Format**:
Return a JSON object with the following structure:
{{
    "lineage_diagram": "text-based diagram showing data flow",
    "field_mappings": [
        {{
            "source_table": "source_table_name",
            "source_field": "source_field_name",
            "target_table": "target_table_name",
            "target_field": "target_field_name",
            "transformations": ["list of transformations applied"]
        }}
    ],
    "table_relationships": [
        {{
            "source_table": "table_name_1",
            "target_table": "table_name_2",
            "relationship_type": "join|lookup|union",
            "join_condition": "condition if applicable"
        }}
    ]
}}

Create a detailed data lineage that a data steward would find valuable for governance purposes.
"""

    def _build_sql_generation_prompt(self, context: ConversionContext, analysis: str, lineage: str) -> str:
        """Build prompt for SQL code generation with enhanced Spark SQL guidance"""
        db_specific_guidance = self._get_database_specific_guidance(context.target_database)
        
        # Special handling for stored procedures
        if context.etl_tool == ETLToolType.STORED_PROCEDURE:
            return self._build_stored_procedure_conversion_prompt(context, analysis, lineage, db_specific_guidance)
        elif context.etl_tool == ETLToolType.SQL_SCRIPT:
            return self._build_sql_script_conversion_prompt(context, analysis, lineage, db_specific_guidance)
        
        # Standard ETL tool prompt
        return f"""
You are an expert SQL developer generating optimized SQL code from ETL logic.

**Task**: Convert the analyzed ETL logic to equivalent SQL code for {context.target_database.value}.

**Analysis Results**:
{analysis}

**Data Lineage Information**:
{lineage}

**Original ETL Code**:
```
{context.source_code}
```

{db_specific_guidance}

**Requirements**:
1. Generate functionally equivalent SQL code
2. Maintain all business rules and data transformations
3. Include proper error handling and data quality checks
4. Use {context.target_database.value}-specific SQL features and functions
5. Optimize for performance with proper join strategies
6. Include comprehensive comments explaining the logic

**SQL Best Practices to Follow**:
- Use CTEs (Common Table Expressions) for complex transformations
- Implement proper NULL handling
- Use appropriate data types for {context.target_database.value}
- Structure code for readability and maintainability
- Add performance optimization hints where applicable

**Output Format**:
Generate clean, production-ready SQL code with:
1. Header comments explaining the conversion
2. Well-structured CTEs for complex logic
3. Inline comments for business rules
4. Final SELECT or INSERT/UPDATE statements

Focus on correctness, performance, and maintainability.
"""

    def _build_stored_procedure_conversion_prompt(self, context: ConversionContext, analysis: str, lineage: str, db_specific_guidance: str) -> str:
        """Build specialized prompt for converting stored procedures to SQL"""
        return f"""
You are an expert SQL developer converting procedural database code to set-based SQL operations.

**Task**: Convert the analyzed stored procedure logic to equivalent SQL code for {context.target_database.value}.

**Analysis Results**:
{analysis}

**Data Lineage Information**:
{lineage}

**Original Stored Procedure Code**:
```sql
{context.source_code}
```

{db_specific_guidance}

**Conversion Requirements**:
1. Convert procedural logic to set-based SQL operations
2. Replace cursors with set-based alternatives (window functions, CTEs, etc.)
3. Maintain all business rules and data transformations
4. Replace procedural error handling with SQL error handling when possible
5. Use {context.target_database.value}-specific features effectively
6. Optimize for performance with proper join strategies

**Key Conversion Patterns**:
- Replace cursors with window functions for row-by-row operations
- Use recursive CTEs for iterative logic
- Convert procedural IF/ELSE logic to CASE expressions
- Use MERGE statements for complex UPDATE/INSERT operations
- Replace temporary tables with CTEs where appropriate

**Output Format**:
Generate clean, production-ready SQL code with:
1. Header comments explaining the conversion strategy
2. Well-structured CTEs for complex logic
3. Clear documentation for how procedural logic was converted
4. Inline comments for business rules
5. Final SQL statements properly sequenced

Make sure your solution is functionally equivalent to the original stored procedure.
For any procedural logic that cannot be directly converted to SQL, explain the limitations and propose alternatives.
"""

    def _build_sql_script_conversion_prompt(self, context: ConversionContext, analysis: str, lineage: str, db_specific_guidance: str) -> str:
        """Build specialized prompt for converting SQL scripts to standardized SQL"""
        return f"""
You are an expert SQL developer standardizing SQL scripts for optimal performance.

**Task**: Convert the analyzed SQL script to equivalent standardized SQL code for {context.target_database.value}.

**Analysis Results**:
{analysis}

**Data Lineage Information**:
{lineage}

**Original SQL Script**:
```sql
{context.source_code}
```

{db_specific_guidance}

**Conversion Requirements**:
1. Standardize dialect-specific syntax to {context.target_database.value} syntax
2. Optimize the query flow and execution plan
3. Replace inefficient patterns with set-based alternatives
4. Maintain all business rules and data transformations
5. Optimize temporary table usage or replace with CTEs
6. Structure code for improved maintainability

**Key Conversion Patterns**:
- Replace multiple single-statement operations with set-based operations
- Use CTEs for improved readability and optimization
- Optimize JOIN strategies based on {context.target_database.value} capabilities
- Standardize function calls to use {context.target_database.value} equivalents
- Improve indexing hints or partitioning strategies if applicable

**Output Format**:
Generate clean, production-ready SQL code with:
1. Header comments explaining the conversion strategy
2. Well-structured CTEs for complex logic
3. Optimized query flow that maintains the original execution sequence where needed
4. Inline comments explaining dialect-specific conversions
5. Final SQL statements properly sequenced

Focus on performance, standards compliance, and maintainability while preserving the original functionality.
"""

    def _build_optimization_prompt(self, sql_code: str, target_db: str) -> str:
        """Build prompt for database-specific optimization with emphasis on Spark SQL if applicable"""
        db_specific_guidance = ""
        
        if target_db.lower() == "spark_sql":
            db_specific_guidance = """
**Spark SQL Specific Optimizations**:
- Use DataFrame operations when possible (select, filter, groupBy)
- Leverage Spark SQL functions (from_json, to_json, explode)
- Use window functions for analytics (rank, dense_rank, lag, lead)
- Apply proper partitioning strategies
- Use broadcast joins for small tables
- Consider caching for frequently used data
- Apply predicate pushdown patterns
- Use appropriate Spark SQL data types
"""
        
        return f"""
You are a database performance expert optimizing SQL for {target_db}.

**Task**: Optimize the following SQL code specifically for {target_db}, focusing on:

1. **Database-specific functions**: Use {target_db} native functions
2. **Performance optimization**: Query structure, indexing, partitioning
3. **Best practices**: {target_db} coding standards and conventions
4. **Resource efficiency**: Memory usage, I/O optimization
5. **Maintainability**: Code organization and documentation

{db_specific_guidance}

**SQL Code to Optimize**:
```sql
{sql_code}
```

**Output**: 
Return the optimized SQL code with:
1. Performance improvements highlighted in comments
2. Database-specific functions used
3. Optimization rationale explained
4. Any indexing or partitioning recommendations

Ensure the code is production-ready for {target_db}.
"""

    def _build_documentation_prompt(self, context: ConversionContext, sql_code: str, 
                                  analysis_result: str, lineage_result: str) -> str:
        """Build prompt for generating comprehensive documentation"""
        return f"""
You are a technical documentation expert creating comprehensive documentation for ETL-to-SQL conversion.

**Task**: Generate detailed documentation for the converted SQL code.

**Original ETL Tool**: {context.etl_tool.value}
**Target Database**: {context.target_database.value}

**Analysis Results**:
{analysis_result}

**Data Lineage Information**:
{lineage_result}

**Generated SQL Code**:
```sql
{sql_code}
```

**Documentation Requirements**:
1. **Executive Summary**: High-level overview of the conversion
2. **Technical Specifications**: Detailed technical details
3. **Business Logic Mapping**: How ETL logic maps to SQL
4. **Performance Considerations**: Expected performance characteristics
5. **Data Lineage Diagram**: Visual representation of data flow
6. **Maintenance Guide**: How to maintain and modify the code
7. **Testing Recommendations**: How to validate the conversion
8. **Deployment Notes**: Deployment considerations

**Output Format**: Markdown documentation with clear sections and examples.

Generate comprehensive, professional documentation suitable for both technical teams and data stewards.
"""

    def _build_data_quality_prompt(self, context: ConversionContext, sql_code: str) -> str:
        """Build prompt for generating data quality checks"""
        return f"""
You are a data quality expert creating comprehensive data quality checks for SQL.

**Task**: Generate data quality validation SQL statements for the converted ETL code.

**Original ETL Tool**: {context.etl_tool.value}
**Target Database**: {context.target_database.value}

**Generated SQL Code**:
```sql
{sql_code}
```

**Requirements**:
1. Create data quality checks that verify:
   - Referential integrity
   - Data type validation
   - Null checks where appropriate
   - Business rule compliance
   - Duplicate detection
   - Value range validation
   - Pattern matching for formatted fields
2. Format checks as SQL assertions or validation queries
3. Design checks to be run after the main ETL process

**Output Format**:
SQL statements with comments explaining each check's purpose.
Ensure the checks are compatible with {context.target_database.value}.

Generate comprehensive data quality validation SQL that a data steward would value.
"""

    def _get_database_specific_guidance(self, db_type: DatabaseType) -> str:
        """Get database-specific guidance for the target database"""
        if db_type == DatabaseType.SPARK_SQL:
            return """
**Spark SQL Specific Guidance**:
- Leverage Spark SQL's distributed processing capabilities
- Use window functions for analytics operations
- Apply dataframe operations when beneficial
- Utilize Spark's optimization features like caching and partitioning
- Consider broadcast joins for small tables
- Use Spark SQL functions like explode, from_json, to_json for complex types
"""
        elif db_type == DatabaseType.SNOWFLAKE:
            return """
**Snowflake Specific Guidance**:
- Leverage Snowflake's clustering keys
- Use COPY commands for data loading
- Apply time travel for historical data queries
- Utilize semi-structured data functions
- Consider materialized views for performance
"""
        elif db_type == DatabaseType.BIGQUERY:
            return """
**BigQuery Specific Guidance**:
- Leverage partitioning and clustering
- Use nested and repeated fields appropriately
- Apply BigQuery ML for machine learning
- Consider wildcard tables for time-partitioned data
- Use approximate aggregations for large datasets
"""
        else:
            # Generic ANSI SQL guidance
            return """
**SQL Best Practices**:
- Use Common Table Expressions (CTEs) for readability
- Apply proper indexing strategies
- Consider materialized views for complex calculations
- Use ANSI SQL standard functions when possible
- Structure queries for query optimizer efficiency
"""

    async def _call_llm_with_retry(self, prompt: str, task_type: str, retry_count: int = None) -> str:
        """Call LLM with retry logic"""
        if retry_count is None:
            retry_count = self._retry_count
            
        last_exception = None
        
        for attempt in range(1, retry_count + 1):
            try:
                if self.config.provider == LLMProvider.ANTHROPIC_CLAUDE:
                    response = await self._call_anthropic(prompt)
                elif self.config.provider == LLMProvider.OPENAI_GPT4:
                    response = await self._call_openai(prompt)
                elif self.config.provider == LLMProvider.AZURE_OPENAI:
                    response = await self._call_azure_openai(prompt)
                elif self.config.provider == LLMProvider.GEMINI:
                    response = await self._call_gemini(prompt)
                else:
                    raise ValueError(f"LLM provider not implemented: {self.config.provider}")
                
                logger.info(f"LLM call successful for task: {task_type} (attempt {attempt})")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call failed for {task_type} (attempt {attempt}/{retry_count}): {e}")
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        logger.error(f"All {retry_count} attempts failed for task: {task_type}")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Failed to complete LLM call for task: {task_type}")

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT API"""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{
                "role": "system",
                "content": "You are an expert data engineer specializing in ETL to SQL conversion, with deep knowledge of data lineage, Spark SQL, and data quality principles."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

    async def _call_azure_openai(self, prompt: str) -> str:
        """Call Azure OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{
                "role": "system",
                "content": "You are an expert data engineer specializing in ETL to SQL conversion, with deep knowledge of data lineage, Spark SQL, and data quality principles."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API via LangChain Google Generative AI"""
        # LangChain's ChatGoogleGenerativeAI is synchronous, so run in thread executor
        loop = asyncio.get_event_loop()
        def run_sync():
            return self.client.invoke([
                {"role": "system", "content": "You are an expert data engineer specializing in ETL to SQL conversion, with deep knowledge of data lineage, Spark SQL, and data quality principles."},
                {"role": "user", "content": prompt}
            ])
        response = await loop.run_in_executor(None, run_sync)
        # LangChain returns a string directly
        return str(response)

class ETLAnalyzer:
    """Enhanced analyzer for various ETL code formats"""
    
    def analyze(self, etl_code: str, etl_tool_type: ETLToolType) -> Dict[str, Any]:
        """Analyze ETL code based on tool type"""
        try:
            if etl_tool_type == ETLToolType.INFORMATICA:
                return self._analyze_informatica(etl_code)
            elif etl_tool_type == ETLToolType.PYTHON_PANDAS:
                return self._analyze_python_pandas(etl_code)
            elif etl_tool_type == ETLToolType.PYSPARK:
                return self._analyze_pyspark(etl_code)
            elif etl_tool_type == ETLToolType.TALEND:
                return self._analyze_talend(etl_code)
            elif etl_tool_type == ETLToolType.SSIS:
                return self._analyze_ssis(etl_code)
            elif etl_tool_type == ETLToolType.STORED_PROCEDURE:
                return self._analyze_stored_procedure(etl_code)
            elif etl_tool_type == ETLToolType.SQL_SCRIPT:
                return self._analyze_sql_script(etl_code)
            else:
                # Generic analysis
                return self._analyze_generic(etl_code, etl_tool_type)
        except Exception as e:
            logger.error(f"ETL analysis failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "tool_type": etl_tool_type.value,
                "complexity": "unknown"
            }
    
    def _analyze_informatica(self, xml_content: str) -> Dict[str, Any]:
        """Analyze Informatica mapping XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Extract sources
            sources = []
            for source in root.findall(".//SOURCE"):
                source_info = {
                    'name': source.get('NAME', ''),
                    'database_type': source.get('DATABASETYPE', ''),
                    'columns': []
                }
                
                for field in source.findall(".//TRANSFORMFIELD"):
                    source_info['columns'].append({
                        'name': field.get('NAME'),
                        'datatype': field.get('DATATYPE'),
                        'precision': field.get('PRECISION'),
                        'scale': field.get('SCALE')
                    })
                sources.append(source_info)
            
            # Extract transformations
            transformations = []
            for trans in root.findall(".//TRANSFORMATION"):
                trans_info = {
                    'type': trans.get('TYPE'),
                    'name': trans.get('NAME'),
                    'expressions': [],
                    'conditions': []
                }
                
                for expr in trans.findall(".//EXPRESSION"):
                    trans_info['expressions'].append({
                        'port': expr.get('PORT'),
                        'expression': expr.text or ''
                    })
                
                # Extract conditions (join conditions, filter conditions)
                for cond in trans.findall(".//CONDITION"):
                    trans_info['conditions'].append(cond.text or '')
                
                transformations.append(trans_info)
            
            # Analyze complexity
            complexity = self._assess_informatica_complexity(transformations)
            
            return {
                'sources': sources,
                'transformations': transformations,
                'complexity': complexity,
                'tool_type': 'informatica'
            }
        
        except Exception as e:
            logger.error(f"Informatica analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'informatica',
                'complexity': 'unknown'
            }
    
    def _analyze_python_pandas(self, code: str) -> Dict[str, Any]:
        """Analyze Python Pandas ETL code"""
        try:
            tree = ast.parse(code)
            
            imports = []
            dataframes = []
            operations = []
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
            
            # Look for dataframe creations and operations
            for node in ast.walk(tree):
                # DataFrame creation from file
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) 
                    and node.func.attr == 'read_csv' 
                    and hasattr(node.func, 'value') and isinstance(node.func.value, ast.Name) 
                    and node.func.value.id == 'pd'):
                    
                    if len(node.args) > 0:
                        file_path = ast.literal_eval(node.args[0]) if isinstance(node.args[0], ast.Constant) else "unknown"
                        dataframes.append({
                            'source_type': 'csv',
                            'path': file_path
                        })
                
                # DataFrame operations
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ['merge', 'join', 'groupby', 'agg', 'filter', 'apply']):
                    
                    operations.append({
                        'type': node.func.attr,
                        'context': ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                    })
            
            # Determine complexity
            if len([op for op in operations if op['type'] in ['apply', 'transform']]) > 0:
                complexity = 'high'
            elif len([op for op in operations if op['type'] in ['merge', 'join', 'groupby']]) > 0:
                complexity = 'medium'
            else:
                complexity = 'low'
                
            return {
                'imports': imports,
                'dataframes': dataframes,
                'operations': operations,
                'complexity': complexity,
                'tool_type': 'python_pandas'
            }
                
        except Exception as e:
            logger.error(f"Python Pandas analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'python_pandas',
                'complexity': 'unknown'
            }
    
    def _analyze_pyspark(self, code: str) -> Dict[str, Any]:
        """Analyze PySpark ETL code"""
        try:
            tree = ast.parse(code)
            
            imports = []
            spark_operations = []
            sql_queries = []
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
            
            # Look for Spark operations and SQL
            for node in ast.walk(tree):
                # Spark operations
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['read', 'write', 'createOrReplaceTempView', 'sql']:
                        spark_operations.append({
                            'type': node.func.attr,
                            'context': ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                        })
                    
                # Look for SQL strings in spark.sql calls
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) 
                    and node.func.attr == 'sql' and len(node.args) > 0 
                    and isinstance(node.args[0], ast.Constant)):
                    
                    sql_queries.append(node.args[0].value)
            
            return {
                'imports': imports,
                'spark_operations': spark_operations,
                'sql_queries': sql_queries,
                'complexity': 'medium' if len(sql_queries) > 0 else 'high',
                'tool_type': 'pyspark'
            }
                
        except Exception as e:
            logger.error(f"PySpark analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'pyspark',
                'complexity': 'unknown'
            }
    
    def _analyze_talend(self, xml_content: str) -> Dict[str, Any]:
        """Basic Talend job XML analysis"""
        try:
            root = ET.fromstring(xml_content)
            
            components = []
            connections = []
            
            # Extract components
            for component in root.findall(".//node"):
                comp_info = {
                    'name': component.get('componentName', ''),
                    'label': component.get('label', ''),
                    'component_type': component.get('componentType', ''),
                    'parameters': []
                }
                
                # Extract parameters
                for param in component.findall(".//elementParameter"):
                    if param.get('field') == 'TEXT':
                        comp_info['parameters'].append({
                            'name': param.get('name'),
                            'value': param.get('value')
                        })
                
                components.append(comp_info)
            
            # Extract connections
            for connection in root.findall(".//connection"):
                conn_info = {
                    'source': connection.get('source'),
                    'target': connection.get('target'),
                    'connector_type': connection.get('connectorName', '')
                }
                connections.append(conn_info)
            
            # Determine complexity
            sql_components = [c for c in components if 'SQL' in c['component_type']]
            complexity = 'high' if len(sql_components) > 3 else 'medium' if len(sql_components) > 0 else 'low'
            
            return {
                'components': components,
                'connections': connections,
                'complexity': complexity,
                'tool_type': 'talend'
            }
                
        except Exception as e:
            logger.error(f"Talend analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'talend',
                'complexity': 'unknown'
            }
    
    def _analyze_ssis(self, xml_content: str) -> Dict[str, Any]:
        """Basic SSIS package XML analysis"""
        try:
            root = ET.fromstring(xml_content)
            
            # SSIS namespace
            ns = {'DTS': 'www.microsoft.com/SqlServer/Dts'}
            
            tasks = []
            data_flows = []
            
            # Extract tasks
            for task in root.findall(".//DTS:Executable", ns):
                task_info = {
                    'name': task.get('{' + ns['DTS'] + '}ObjectName', ''),
                    'task_type': task.get('{' + ns['DTS'] + '}CreationName', ''),
                    'properties': []
                }
                
                # Extract properties
                for prop in task.findall(".//DTS:Property", ns):
                    task_info['properties'].append({
                        'name': prop.get('{' + ns['DTS'] + '}Name'),
                        'value': prop.text
                    })
                
                tasks.append(task_info)
                
                # Look for data flow tasks
                if 'Pipeline' in task_info['task_type']:
                    data_flow_components = []
                    
                    for component in task.findall(".//components/component", ns):
                        comp_info = {
                            'name': component.get('name', ''),
                            'component_type': component.get('componentClassID', '')
                        }
                        data_flow_components.append(comp_info)
                    
                    data_flows.append({
                        'task_name': task_info['name'],
                        'components': data_flow_components
                    })
            
            # Determine complexity based on number of components
            total_components = sum(len(df['components']) for df in data_flows)
            complexity = 'high' if total_components > 10 else 'medium' if total_components > 5 else 'low'
            
            return {
                'tasks': tasks,
                'data_flows': data_flows,
                'complexity': complexity,
                'tool_type': 'ssis'
            }
                
        except Exception as e:
            logger.error(f"SSIS analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'ssis',
                'complexity': 'unknown'
            }
    
    def _analyze_generic(self, code: str, etl_tool_type: ETLToolType) -> Dict[str, Any]:
        """Generic analysis when specific parser isn't available"""
        # Simple metrics to determine complexity
        lines = code.splitlines()
        
        metrics = {
            'line_count': len(lines),
            'character_count': len(code),
            'avg_line_length': len(code) / max(len(lines), 1),
            'contains_joins': 'join' in code.lower(),
            'contains_aggregations': any(agg in code.lower() for agg in ['sum(', 'avg(', 'count(', 'min(', 'max(']),
            'contains_window_functions': 'over(' in code.lower() or 'partition by' in code.lower()
        }
        
        # Rough complexity assessment
        if metrics['line_count'] > 200 or metrics['contains_window_functions']:
            complexity = 'high'
        elif metrics['line_count'] > 100 or metrics['contains_joins'] or metrics['contains_aggregations']:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return {
            'metrics': metrics,
            'complexity': complexity,
            'tool_type': etl_tool_type.value
        }
    
    def _analyze_stored_procedure(self, code: str) -> Dict[str, Any]:
        """Analyze stored procedure code for ETL operations
        
        Parses stored procedures from various databases (SQL Server, Oracle, PostgreSQL, MySQL)
        and identifies ETL patterns such as data movement, transformations, and loading logic.
        """
        try:
            # Normalize line endings and remove extra whitespace
            code = code.replace('\r\n', '\n').strip()
            
            # Determine the likely database dialect
            dialect = self._detect_sp_dialect(code)
            
            # Extract procedure name and parameters
            procedure_name = self._extract_procedure_name(code, dialect)
            parameters = self._extract_procedure_parameters(code, dialect)
            
            # Extract statements and blocks
            statements = self._extract_sp_statements(code, dialect)
            
            # Find data sources and targets
            sources = self._extract_sp_data_sources(code, statements, dialect)
            targets = self._extract_sp_data_targets(code, statements, dialect)
            
            # Identify ETL operations
            operations = []
            for stmt in statements:
                if self._is_etl_operation(stmt, dialect):
                    operations.append({
                        'operation_type': self._classify_etl_operation(stmt, dialect),
                        'statement': stmt,
                        'complexity': self._assess_statement_complexity(stmt, dialect)
                    })
            
            # Determine overall complexity
            complex_ops = [op for op in operations if op['complexity'] == 'high']
            medium_ops = [op for op in operations if op['complexity'] == 'medium']
            
            if len(complex_ops) > 2 or len(operations) > 10:
                complexity = 'high'
            elif len(complex_ops) > 0 or len(medium_ops) > 3:
                complexity = 'medium'
            else:
                complexity = 'low'
            
            # Check for cursor usage which often indicates complex ETL
            cursor_usage = self._detect_cursor_usage(code, dialect)
            if cursor_usage['has_cursors'] and cursor_usage['cursor_count'] > 1:
                complexity = 'high'
            
            return {
                'procedure_name': procedure_name,
                'parameters': parameters,
                'dialect': dialect,
                'sources': sources,
                'targets': targets,
                'operations': operations,
                'cursor_usage': cursor_usage,
                'complexity': complexity,
                'tool_type': 'stored_procedure'
            }
                
        except Exception as e:
            logger.error(f"Stored procedure analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'stored_procedure',
                'complexity': 'unknown'
            }
    
    def _analyze_sql_script(self, code: str) -> Dict[str, Any]:
        """Analyze SQL script for ETL operations
        
        Examines SQL scripts that may contain multiple statements, temporary tables,
        and ETL logic without being wrapped in a formal stored procedure.
        """
        try:
            # Normalize line endings and remove extra whitespace
            code = code.replace('\r\n', '\n').strip()
            
            # Determine the likely database dialect
            dialect = self._detect_sql_dialect(code)
            
            # Split into statements
            statements = self._split_sql_statements(code, dialect)
            
            # Find data sources and targets
            sources = self._extract_script_data_sources(statements, dialect)
            targets = self._extract_script_data_targets(statements, dialect)
            
            # Identify transformations
            transformations = []
            for stmt in statements:
                if self._is_transformation_statement(stmt, dialect):
                    transformations.append({
                        'type': self._classify_transformation(stmt, dialect),
                        'statement': stmt,
                        'complexity': self._assess_statement_complexity(stmt, dialect)
                    })
            
            # Analyze workflow structure
            workflow = self._analyze_sql_workflow(statements, dialect)
            
            # Detect temporary tables (often used in ETL scripts)
            temp_tables = self._detect_temp_tables(statements, dialect)
            
            # Determine overall complexity
            if len(transformations) > 8 or len(statements) > 15 or len(temp_tables) > 3:
                complexity = 'high'
            elif len(transformations) > 4 or len(statements) > 8 or len(temp_tables) > 1:
                complexity = 'medium'
            else:
                complexity = 'low'
            
            return {
                'dialect': dialect,
                'statement_count': len(statements),
                'sources': sources,
                'targets': targets,
                'transformations': transformations,
                'temp_tables': temp_tables,
                'workflow': workflow,
                'complexity': complexity,
                'tool_type': 'sql_script'
            }
                
        except Exception as e:
            logger.error(f"SQL script analysis error: {e}", exc_info=True)
            return {
                'error': str(e),
                'tool_type': 'sql_script',
                'complexity': 'unknown'
            }
    
    def _detect_sp_dialect(self, code: str) -> str:
        """Detect the stored procedure dialect based on syntax patterns"""
        code_lower = code.lower()
        
        # SQL Server patterns
        if ('create procedure' in code_lower or 'create proc' in code_lower) and \
           ('begin try' in code_lower or 'declare @' in code_lower or 'set nocount on' in code_lower):
            return 'tsql'
        
        # Oracle patterns
        elif ('create or replace procedure' in code_lower or 'create procedure' in code_lower) and \
             ('begin' in code_lower and 'end;' in code_lower) and \
             ('dbms_' in code_lower or 'v_' in code_lower or 'l_' in code_lower):
            return 'plsql'
        
        # PostgreSQL patterns
        elif ('create or replace function' in code_lower and 'returns ' in code_lower) and \
             ('begin' in code_lower and 'end;' in code_lower) and \
             ('$$' in code_lower or 'language plpgsql' in code_lower):
            return 'plpgsql'
        
        # MySQL patterns
        elif ('create procedure' in code_lower or 'create function' in code_lower) and \
             ('begin' in code_lower and 'end' in code_lower) and \
             ('declare' in code_lower or 'delimiter' in code_lower):
            return 'mysql'
        
        # Default
        else:
            # Make best guess based on most common patterns
            if 'end;' in code_lower and 'declare' in code_lower:
                return 'plsql'
            elif '@' in code_lower:
                return 'tsql'
            else:
                return 'generic_sql'
    
    def _detect_sql_dialect(self, code: str) -> str:
        """Detect the SQL dialect based on syntax patterns"""
        code_lower = code.lower()
        
        # Spark SQL patterns
        if ('spark' in code_lower or 'hive' in code_lower) and \
           ('from_json' in code_lower or 'to_json' in code_lower or 'explode' in code_lower):
            return 'spark_sql'
        
        # Snowflake patterns
        elif ('variant' in code_lower or '$$ ' in code_lower or 
              'flatten(' in code_lower or 'table(split_to_table' in code_lower):
            return 'snowflake'
        
        # BigQuery patterns
        elif ('bq.' in code_lower or 'bigquery' in code_lower or 
              'struct(' in code_lower or 'unnest(' in code_lower):
            return 'bigquery'
        
        # SQL Server patterns
        elif ('isnull(' in code_lower or 'convert(' in code_lower or 'top ' in code_lower):
            return 'tsql'
        
        # Oracle patterns
        elif ('nvl(' in code_lower or 'rownum' in code_lower or 'connect by' in code_lower):
            return 'oracle'
        
        # PostgreSQL patterns
        elif ('::' in code_lower or 'ilike' in code_lower):
            return 'postgresql'
        
        # MySQL patterns
        elif ('ifnull(' in code_lower or 'limit ' in code_lower and 'offset' in code_lower):
            return 'mysql'
        
        # Default to generic SQL
        else:
            return 'ansi_sql'
    
    def _extract_procedure_name(self, code: str, dialect: str) -> str:
        """Extract the stored procedure name based on dialect"""
        code_lower = code.lower()
        
        if dialect == 'tsql':
            match = re.search(r'create\s+(proc|procedure)\s+(\w+\.)?(\w+)', code_lower)
            if match:
                return match.group(3)
        elif dialect in ['plsql', 'plpgsql']:
            match = re.search(r'create\s+(?:or\s+replace\s+)?(?:procedure|function)\s+(\w+\.)?(\w+)', code_lower)
            if match:
                return match.group(2)
        elif dialect == 'mysql':
            match = re.search(r'create\s+(?:procedure|function)\s+(\w+\.)?(\w+)', code_lower)
            if match:
                return match.group(2)
                
        # If we can't extract a specific name
        return "unknown_procedure"
    
    def _extract_procedure_parameters(self, code: str, dialect: str) -> List[Dict[str, str]]:
        """Extract procedure parameters based on dialect"""
        params = []
        code_lower = code.lower()
        
        # Extract everything between parentheses after procedure/function name
        pattern = r'create\s+(?:or\s+replace\s+)?(?:proc|procedure|function)\s+(?:\w+\.)?(\w+)\s*\((.*?)\)'
        match = re.search(pattern, code_lower, re.DOTALL)
        
        if not match:
            return params
            
        param_text = match.group(2).strip()
        
        # If no parameters
        if not param_text:
            return params
            
        # Handle different parameter syntax based on dialect
        if dialect == 'tsql':
            # T-SQL: @param_name data_type
            param_matches = re.finditer(r'@(\w+)\s+([^,=]+?)(?:=\s*[^,]+)?(?:,|$)', param_text)
            for m in param_matches:
                params.append({
                    'name': '@' + m.group(1),
                    'data_type': m.group(2).strip(),
                    'mode': 'IN'  # Default for SQL Server
                })
                
        elif dialect in ['plsql', 'mysql', 'plpgsql']:
            # Oracle/MySQL/PostgreSQL: param_name IN/OUT data_type
            param_parts = re.split(r',\s*', param_text)
            for part in param_parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Try to match: param_name [IN/OUT/INOUT] data_type
                match = re.search(r'(\w+)(?:\s+(IN|OUT|IN\s+OUT|INOUT))?\s+([^,]+)', part, re.IGNORECASE)
                if match:
                    mode = (match.group(2) or 'IN').upper()
                    params.append({
                        'name': match.group(1),
                        'data_type': match.group(3).strip(),
                        'mode': 'INOUT' if mode == 'IN OUT' else mode
                    })
        
        return params
    
    def _extract_sp_statements(self, code: str, dialect: str) -> List[str]:
        """Extract individual statements from stored procedure code"""
        # Find the procedure body
        if dialect == 'tsql':
            # For SQL Server, find content between AS/BEGIN and END
            match = re.search(r'(?:as|AS)\s+(?:begin|BEGIN)(.*?)(?:end|END)(?:\s+(?:catch|CATCH).*?(?:end|END))?', 
                             code, re.DOTALL)
        elif dialect in ['plsql', 'plpgsql', 'mysql']:
            # For Oracle/PostgreSQL/MySQL find content between BEGIN and END
            match = re.search(r'(?:begin|BEGIN)(.*?)(?:end|END)', code, re.DOTALL)
        else:
            # Generic approach
            match = re.search(r'(?:as|AS|begin|BEGIN)(.*?)(?:end|END)', code, re.DOTALL)
            
        if not match:
            # If we can't identify the body clearly, use the whole code
            body = code
        else:
            body = match.group(1)
            
        # Split into statements based on dialect
        statements = []
        
        if dialect == 'tsql':
            # Split on semicolons but be careful with control flow statements
            current_stmt = ""
            lines = body.split('\n')
            
            for line in lines:
                line_stripped = line.strip()
                current_stmt += line + "\n"
                
                # Skip empty lines and comments
                if not line_stripped or line_stripped.startswith('--'):
                    continue
                    
                # If we have a complete statement
                if line_stripped.endswith(';'):
                    if current_stmt.strip():
                        statements.append(current_stmt.strip())
                    current_stmt = ""
                    
                # Some T-SQL statements don't require semicolons
                elif re.match(r'^\s*(if|while|begin|end)\b', line_stripped, re.IGNORECASE):
                    if current_stmt.strip():
                        statements.append(current_stmt.strip())
                    current_stmt = ""
            
            # Add any remaining statement
            if current_stmt.strip():
                statements.append(current_stmt.strip())
                
        elif dialect in ['plsql', 'plpgsql', 'mysql']:
            # For Oracle/PostgreSQL/MySQL, split on semicolons but respect PL blocks
            current_stmt = ""
            in_block = False
            lines = body.split('\n')
            
            for line in lines:
                line_stripped = line.strip().lower()
                current_stmt += line + "\n"
                
                # Skip empty lines and comments
                if not line_stripped or line_stripped.startswith('--'):
                    continue
                
                # Check for block start/end
                if re.search(r'\b(begin|loop|if)\b', line_stripped):
                    in_block = True
                elif re.search(r'\b(end)\b', line_stripped):
                    in_block = False
                    
                # If we have a complete statement and not in a block
                if line_stripped.endswith(';') and not in_block:
                    if current_stmt.strip():
                        statements.append(current_stmt.strip())
                    current_stmt = ""
            
            # Add any remaining statement
            if current_stmt.strip():
                statements.append(current_stmt.strip())
        else:
            # Generic approach - split on semicolons
            raw_statements = re.split(r';\s*(?=\w)', body)
            statements = [stmt.strip() for stmt in raw_statements if stmt.strip()]
        
        return statements
    
    def _extract_sp_data_sources(self, code: str, statements: List[str], dialect: str) -> List[Dict[str, Any]]:
        """Extract data sources from stored procedure code"""
        sources = []
        
        # Regular expressions to match SELECT statements
        select_patterns = [
            # Basic pattern: SELECT ... FROM table
            r'(?:select|SELECT).*?(?:from|FROM)\s+([^\s\(]+)',
            # Join pattern: JOIN table
            r'(?:join|JOIN)\s+([^\s\(]+)',
            # Insert-Select pattern: INSERT INTO target SELECT ... FROM source
            r'(?:insert\s+into|INSERT\s+INTO).*?(?:select|SELECT).*?(?:from|FROM)\s+([^\s\(]+)'
        ]
        
        # Extract table names from statements
        table_names = set()
        for stmt in statements:
            for pattern in select_patterns:
                matches = re.finditer(pattern, stmt)
                for match in matches:
                    table_name = match.group(1).strip()
                    # Clean up the table name (remove aliases, etc.)
                    table_name = re.sub(r'\s+as\s+\w+', '', table_name, flags=re.IGNORECASE)
                    table_name = re.sub(r'\s+\w+$', '', table_name)  # Remove trailing alias
                    table_name = table_name.strip('[]"\'`')  # Remove quoting characters
                    
                    if table_name and table_name.lower() not in ('dual', 'sysdate'):  # Exclude non-tables
                        table_names.add(table_name)
        
        # Convert table names to source objects
        for table in table_names:
            # Skip common temp table prefixes
            if table.startswith('#') or table.startswith('@'):
                continue
                
            sources.append({
                'name': table,
                'type': 'table',
                'schema': self._extract_schema_from_table(table)
            })
        
        return sources
    
    def _extract_sp_data_targets(self, code: str, statements: List[str], dialect: str) -> List[Dict[str, Any]]:
        """Extract data targets from stored procedure code"""
        targets = []
        
        # Regular expressions to match target tables
        target_patterns = [
            # INSERT pattern
            r'(?:insert\s+into|INSERT\s+INTO)\s+([^\s\(]+)',
            # UPDATE pattern
            r'(?:update|UPDATE)\s+([^\s\(]+)',
            # MERGE pattern
            r'(?:merge\s+into|MERGE\s+INTO)\s+([^\s\(]+)',
            # DELETE pattern
            r'(?:delete\s+from|DELETE\s+FROM)\s+([^\s\(]+)',
            # CREATE TABLE pattern
            r'(?:create\s+table|CREATE\s+TABLE)\s+([^\s\(]+)'
        ]
        
        # Extract table names from statements
        table_names = set()
        for stmt in statements:
            for pattern in target_patterns:
                matches = re.finditer(pattern, stmt)
                for match in matches:
                    table_name = match.group(1).strip()
                    # Clean up the table name
                    table_name = table_name.strip('[]"\'`')  # Remove quoting characters
                    
                    if table_name:
                        table_names.add(table_name)
        
        # Convert table names to target objects
        for table in table_names:
            # Include temp tables here
            targets.append({
                'name': table,
                'type': 'table' if not (table.startswith('#') or table.startswith('@')) else 'temp_table',
                'schema': self._extract_schema_from_table(table) if not (table.startswith('#') or table.startswith('@')) else None
            })
        
        return targets
    
    def _extract_schema_from_table(self, table_name: str) -> Optional[str]:
        """Extract schema name from fully qualified table name"""
        parts = table_name.split('.')
        if len(parts) > 1:
            return parts[0]
        return None
    
    def _is_etl_operation(self, statement: str, dialect: str) -> bool:
        """Check if a statement contains ETL operations"""
        # Check for data movement patterns
        stmt_lower = statement.lower()
        
        # Data movement operations
        if any(op in stmt_lower for op in ['insert', 'update', 'merge', 'delete', 'truncate']):
            return True
            
        # Data transformations in SELECT
        if 'select' in stmt_lower and any(func in stmt_lower for func in 
                                          ['sum(', 'avg(', 'count(', 'max(', 'min(', 'cast(', 'convert(',
                                           'substring(', 'concat(', 'replace(', 'case when']):
            return True
            
        # Joins or complex queries
        if 'select' in stmt_lower and any(join_type in stmt_lower for join_type in 
                                         [' join ', ' inner join ', ' left join ', ' right join ', ' full join ']):
            return True
            
        # Table creation or modification
        if any(ddl in stmt_lower for ddl in ['create table', 'alter table', 'drop table']):
            return True
            
        return False
    
    def _classify_etl_operation(self, statement: str, dialect: str) -> str:
        """Classify the type of ETL operation in a statement"""
        stmt_lower = statement.lower()
        
        # Extract operation category
        if 'insert' in stmt_lower and 'select' in stmt_lower:
            return 'insert_select'
        elif 'insert' in stmt_lower:
            return 'insert'
        elif 'update' in stmt_lower:
            return 'update'
        elif 'merge' in stmt_lower:
            return 'merge'
        elif 'delete' in stmt_lower:
            return 'delete'
        elif 'truncate' in stmt_lower:
            return 'truncate'
        elif 'create table' in stmt_lower:
            return 'create_table'
        elif 'alter table' in stmt_lower:
            return 'alter_table'
        elif 'drop table' in stmt_lower:
            return 'drop_table'
        elif 'select' in stmt_lower:
            # Determine the type of SELECT
            if ' join ' in stmt_lower:
                return 'join_query'
            elif any(agg in stmt_lower for agg in ['sum(', 'avg(', 'count(', 'max(', 'min(']):
                return 'aggregate_query'
            elif 'group by' in stmt_lower:
                return 'group_by_query'
            elif 'order by' in stmt_lower:
                return 'sorted_query'
            else:
                return 'select_query'
        else:
            return 'other'
    
    def _assess_statement_complexity(self, statement: str, dialect: str) -> str:
        """Assess the complexity of a SQL statement"""
        stmt_lower = statement.lower()
        
        # Count various complexity factors
        join_count = len(re.findall(r'\bjoin\b', stmt_lower))
        subquery_count = len(re.findall(r'\(select', stmt_lower))
        condition_count = len(re.findall(r'\bwhere\b', stmt_lower))
        case_count = len(re.findall(r'\bcase\b', stmt_lower))
        aggregation_count = len(re.findall(r'\b(sum|avg|count|max|min)\s*\(', stmt_lower))
        
        # Calculate complexity score
        complexity_score = (
            join_count * 2 + 
            subquery_count * 3 + 
            condition_count + 
            case_count * 2 +
            aggregation_count
        )
        
        # Classify based on score
        if complexity_score > 8:
            return 'high'
        elif complexity_score > 3:
            return 'medium'
        else:
            return 'low'
    
    def _detect_cursor_usage(self, code: str, dialect: str) -> Dict[str, Any]:
        """Detect cursor usage in stored procedures"""
        code_lower = code.lower()
        
        cursor_info = {
            'has_cursors': False,
            'cursor_count': 0,
            'cursor_names': []
        }
        
        # Different cursor patterns based on dialect
        if dialect == 'tsql':
            # SQL Server cursor pattern
            cursor_declarations = re.finditer(r'declare\s+(\w+)\s+cursor\s+for', code_lower)
            for match in cursor_declarations:
                cursor_info['has_cursors'] = True
                cursor_info['cursor_count'] += 1
                cursor_info['cursor_names'].append(match.group(1))
                
        elif dialect == 'plsql':
            # Oracle cursor pattern
            cursor_declarations = re.finditer(r'cursor\s+(\w+)\s+is', code_lower)
            for match in cursor_declarations:
                cursor_info['has_cursors'] = True
                cursor_info['cursor_count'] += 1
                cursor_info['cursor_names'].append(match.group(1))
                
        elif dialect in ['plpgsql', 'mysql']:
            # PostgreSQL/MySQL cursor pattern
            cursor_declarations = re.finditer(r'declare\s+(\w+)\s+cursor\s+for', code_lower)
            for match in cursor_declarations:
                cursor_info['has_cursors'] = True
                cursor_info['cursor_count'] += 1
                cursor_info['cursor_names'].append(match.group(1))
        
        # Check for cursor operations regardless of dialect
        if not cursor_info['has_cursors']:
            cursor_ops = ['open', 'fetch', 'close']
            for op in cursor_ops:
                if re.search(r'\b' + op + r'\s+\w+', code_lower):
                    cursor_info['has_cursors'] = True
                    cursor_info['cursor_count'] += 1
                    break
                    
        return cursor_info
    
    def _split_sql_statements(self, code: str, dialect: str) -> List[str]:
        """Split SQL script into individual statements"""
        # Replace comments with empty strings
        code = re.sub(r'--.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Split on semicolons, but be careful with special cases
        statements = []
        current_stmt = ""
        
        lines = code.split('\n')
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                current_stmt += "\n"
                continue
                
            current_stmt += line + "\n"
            
            # Check if the line ends a statement
            if line_stripped.endswith(';'):
                if current_stmt.strip():
                    statements.append(current_stmt.strip())
                current_stmt = ""
        
        # Add any remaining statement
        if current_stmt.strip():
            statements.append(current_stmt.strip())
            
        return statements
    
    def _extract_script_data_sources(self, statements: List[str], dialect: str) -> List[Dict[str, Any]]:
        """Extract data sources from SQL script statements"""
        sources = []
        
        # Regular expressions to match SELECT statements
        select_patterns = [
            r'(?:from|FROM)\s+([^\s\(,]+)',
            r'(?:join|JOIN)\s+([^\s\(,]+)'
        ]
        
        # Extract table names from statements
        table_names = set()
        for stmt in statements:
            # Only process SELECT statements
            if not re.search(r'^\s*(?:select|SELECT)', stmt):
                continue
                
            for pattern in select_patterns:
                matches = re.finditer(pattern, stmt)
                for match in matches:
                    table_name = match.group(1).strip()
                    # Clean up the table name
                    table_name = re.sub(r'\s+as\s+\w+', '', table_name, flags=re.IGNORECASE)
                    table_name = re.sub(r'\s+\w+$', '', table_name)  # Remove trailing alias
                    table_name = table_name.strip('[]"\'`')  # Remove quoting characters
                    
                    if table_name and table_name.lower() not in ('dual', 'sysdate'):
                        table_names.add(table_name)
        
        # Convert table names to source objects
        for table in table_names:
            # Skip common temp table prefixes
            if table.startswith('#') or table.startswith('@'):
                continue
                
            sources.append({
                'name': table,
                'type': 'table',
                'schema': self._extract_schema_from_table(table)
            })
        
        return sources
    
    def _extract_script_data_targets(self, statements: List[str], dialect: str) -> List[Dict[str, Any]]:
        """Extract data targets from SQL script statements"""
        targets = []
        
        # Regular expressions to match target tables
        target_patterns = [
            r'(?:insert\s+into|INSERT\s+INTO)\s+([^\s\(,]+)',
            r'(?:update|UPDATE)\s+([^\s\(,]+)',
            r'(?:merge\s+into|MERGE\s+INTO)\s+([^\s\(,]+)',
            r'(?:delete\s+from|DELETE\s+FROM)\s+([^\s\(,]+)',
            r'(?:create\s+table|CREATE\s+TABLE)\s+([^\s\(,]+)'
        ]
        
        # Extract table names from statements
        table_names = set()
        for stmt in statements:
            for pattern in target_patterns:
                matches = re.finditer(pattern, stmt)
                for match in matches:
                    table_name = match.group(1).strip()
                    # Clean up the table name
                    table_name = table_name.strip('[]"\'`')  # Remove quoting characters
                    
                    if table_name:
                        table_names.add(table_name)
        
        # Convert table names to target objects
        for table in table_names:
            targets.append({
                'name': table,
                'type': 'table' if not (table.startswith('#') or table.startswith('@')) else 'temp_table',
                'schema': self._extract_schema_from_table(table) if not (table.startswith('#') or table.startswith('@')) else None
            })
        
        return targets
    
    def _is_transformation_statement(self, statement: str, dialect: str) -> bool:
        """Check if a statement contains data transformations"""
        stmt_lower = statement.lower()
        
        # Check for transformation patterns
        if 'select' in stmt_lower and (
            'case' in stmt_lower or
            'cast(' in stmt_lower or
            'convert(' in stmt_lower or
            'sum(' in stmt_lower or
            'avg(' in stmt_lower or
            'join' in stmt_lower or
            'union' in stmt_lower or
            'intersect' in stmt_lower or
            'except' in stmt_lower
        ):
            return True
            
        return False
    
    def _classify_transformation(self, statement: str, dialect: str) -> str:
        """Classify the type of transformation in a SQL statement"""
        stmt_lower = statement.lower()
        
        if 'join' in stmt_lower:
            return 'join'
        elif 'union' in stmt_lower:
            return 'union'
        elif 'intersect' in stmt_lower:
            return 'intersect'
        elif 'except' in stmt_lower:
            return 'except'
        elif 'group by' in stmt_lower:
            return 'aggregation'
        elif 'order by' in stmt_lower:
            return 'sort'
        elif 'case' in stmt_lower:
            return 'conditional'
        elif 'cast(' in stmt_lower or 'convert(' in stmt_lower:
            return 'type_conversion'
        else:
            return 'projection'
    
    def _analyze_sql_workflow(self, statements: List[str], dialect: str) -> Dict[str, Any]:
        """Analyze the workflow structure of a SQL script"""
        workflow = {
            'stages': [],
            'temp_tables_flow': {},
            'transaction_blocks': 0,
            'has_error_handling': False
        }
        
        # Track temporary tables for flow analysis
        temp_tables = {}
        
        # Analyze statement sequence
        current_stage = {
            'statements': [],
            'operation_type': 'unknown'
        }
        
        transaction_level = 0
        
        for i, stmt in enumerate(statements):
            stmt_lower = stmt.lower()
            
            # Check for transaction blocks
            if 'begin transaction' in stmt_lower or 'begin tran' in stmt_lower:
                transaction_level += 1
                workflow['transaction_blocks'] += 1
            elif 'commit transaction' in stmt_lower or 'commit tran' in stmt_lower or 'commit' in stmt_lower:
                transaction_level = max(0, transaction_level - 1)
            
            # Check for error handling
            if 'try' in stmt_lower or 'catch' in stmt_lower or 'exception' in stmt_lower:
                workflow['has_error_handling'] = True
            
            # Detect operation type
            if 'create table' in stmt_lower:
                if current_stage['operation_type'] != 'unknown' and current_stage['statements']:
                    workflow['stages'].append(current_stage)
                    current_stage = {
                        'statements': [stmt],
                        'operation_type': 'table_creation'
                    }
                else:
                    current_stage['statements'].append(stmt)
                    current_stage['operation_type'] = 'table_creation'
                
                # Extract table name for flow tracking
                match = re.search(r'create\s+table\s+([^\s\(]+)', stmt_lower)
                if match:
                    table_name = match.group(1).strip('[]"\'`')
                    temp_tables[table_name] = {'created_at': i, 'used_in': []}
                
            elif 'insert' in stmt_lower or 'update' in stmt_lower:
                if current_stage['operation_type'] not in ('data_loading', 'unknown') and current_stage['statements']:
                    workflow['stages'].append(current_stage)
                    current_stage = {
                        'statements': [stmt],
                        'operation_type': 'data_loading'
                    }
                else:
                    current_stage['statements'].append(stmt)
                    current_stage['operation_type'] = 'data_loading'
                
                # Check for temp table usage
                for table_name in temp_tables:
                    if table_name in stmt_lower:
                        temp_tables[table_name]['used_in'].append(i)
                
            elif 'select' in stmt_lower and 'into' in stmt_lower:
                if current_stage['operation_type'] != 'unknown' and current_stage['statements']:
                    workflow['stages'].append(current_stage)
                    current_stage = {
                        'statements': [stmt],
                        'operation_type': 'data_transformation'
                    }
                else:
                    current_stage['statements'].append(stmt)
                    current_stage['operation_type'] = 'data_transformation'
                
                # Extract target table for flow tracking
                match = re.search(r'into\s+([^\s\(]+)', stmt_lower)
                if match:
                    table_name = match.group(1).strip('[]"\'`')
                    temp_tables[table_name] = {'created_at': i, 'used_in': []}
                
                # Check for source temp table usage
                for table_name in temp_tables:
                    if table_name in stmt_lower and 'into ' + table_name not in stmt_lower:
                        temp_tables[table_name]['used_in'].append(i)
                
            else:
                current_stage['statements'].append(stmt)
                if current_stage['operation_type'] == 'unknown':
                    if 'select' in stmt_lower:
                        current_stage['operation_type'] = 'data_query'
                    elif 'drop' in stmt_lower:
                        current_stage['operation_type'] = 'cleanup'
                
                # Check for temp table usage
                for table_name in temp_tables:
                    if table_name in stmt_lower:
                        temp_tables[table_name]['used_in'].append(i)
        
        # Add the final stage
        if current_stage['statements']:
            workflow['stages'].append(current_stage)
        
        # Build temp table flow graph
        for table_name, info in temp_tables.items():
            workflow['temp_tables_flow'][table_name] = {
                'created_at': info['created_at'],
                'used_in': sorted(set(info['used_in']))
            }
        
        return workflow
    
    def _detect_temp_tables(self, statements: List[str], dialect: str) -> List[Dict[str, Any]]:
        """Detect temporary tables in SQL statements"""
        temp_tables = []
        
        for stmt in statements:
            stmt_lower = stmt.lower()
            
            # Detect temporary table creation
            if 'create' in stmt_lower and 'table' in stmt_lower:
                # Different temp table patterns by dialect
                if dialect == 'tsql':
                    match = re.search(r'create\s+table\s+(#\w+|\[#\w+\])', stmt_lower)
                    if match:
                        temp_tables.append({
                            'name': match.group(1).strip('[]'),
                            'dialect': dialect,
                            'statement': stmt
                        })
                elif dialect == 'oracle':
                    if 'global temporary table' in stmt_lower:
                        match = re.search(r'create\s+global\s+temporary\s+table\s+(\w+)', stmt_lower)
                        if match:
                            temp_tables.append({
                                'name': match.group(1),
                                'dialect': dialect,
                                'statement': stmt
                            })
                elif dialect in ['postgresql', 'plpgsql']:
                    if 'temporary table' in stmt_lower or 'temp table' in stmt_lower:
                        match = re.search(r'create\s+(?:temporary|temp)\s+table\s+(\w+)', stmt_lower)
                        if match:
                            temp_tables.append({
                                'name': match.group(1),
                                'dialect': dialect,
                                'statement': stmt
                            })
                elif dialect in ['mysql', 'spark_sql', 'hive']:
                    if 'temporary table' in stmt_lower:
                        match = re.search(r'create\s+temporary\s+table\s+(\w+)', stmt_lower)
                        if match:
                            temp_tables.append({
                                'name': match.group(1),
                                'dialect': dialect,
                                'statement': stmt
                            })
        
        return temp_tables
    
    def _assess_informatica_complexity(self, transformations: List[Dict]) -> str:
        """Assess complexity of Informatica transformations"""
        complex_types = ['Lookup', 'Aggregator', 'Joiner', 'Union', 'Rank', 'Router']
        complex_count = sum(1 for t in transformations if t.get('type') in complex_types)
        
        if complex_count > 3:
            return 'high'
        elif complex_count > 1:
            return 'medium'
        else:
            return 'low'

class AdvancedETLConverter:
    """Main ETL converter with enhanced features for data stewards"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the converter"""
        if llm_config is None:
            # Try to initialize from environment variables
            llm_config = LLMConfig.from_env()
            
        self.llm_orchestrator = LLMOrchestrator(llm_config)
        self.analyzer = ETLAnalyzer()
        self._history = []
        self.conversion_metrics = {
            'successful_conversions': 0,
            'failed_conversions': 0,
            'average_time': 0.0
        }
    
    async def convert(self,
                     etl_code: str,
                     etl_tool: Union[str, ETLToolType],
                     target_database: Union[str, DatabaseType] = DatabaseType.SPARK_SQL,
                     schema_info: Optional[Dict[str, Any]] = None,
                     business_rules: Optional[List[str]] = None,
                     include_lineage: bool = True,
                     include_data_quality: bool = True) -> ConversionResult:
        """
        Main conversion method for ETL to SQL
        
        Args:
            etl_code: The ETL code to convert
            etl_tool: The ETL tool type (informatica, python_pandas, etc.)
            target_database: The target database type (default: spark_sql)
            schema_info: Optional schema information
            business_rules: Optional business rules
            include_lineage: Whether to include data lineage (default: True)
            include_data_quality: Whether to include data quality checks (default: True)
            
        Returns:
            ConversionResult object with conversion results
        """
        start_time = datetime.now()
        
        try:
            # Convert string enums to proper Enum types if needed
            if isinstance(etl_tool, str):
                etl_tool = ETLToolType(etl_tool)
                
            if isinstance(target_database, str):
                target_database = DatabaseType(target_database)
            
            # Step 1: Analyze ETL code structure
            analysis = self.analyzer.analyze(etl_code, etl_tool)
            
            # Step 2: Build conversion context
            context = ConversionContext(
                etl_tool=etl_tool,
                source_code=etl_code,
                target_database=target_database,
                schema_info=schema_info or {},
                business_rules=business_rules or [],
                data_lineage={} if include_lineage else None,
                data_quality_rules=[] if include_data_quality else None,
                execution_context={'analysis': analysis}
            )
            
            # Step 3: Let LLM do the intelligent conversion
            result = await self.llm_orchestrator.convert_etl_to_sql(context)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(result.success, execution_time)
            
            # Store in history
            self._add_to_history(etl_tool.value, target_database.value, result.success)
            
            return result
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, execution_time)
            
            return ConversionResult.from_exception(e)
    
    def _update_metrics(self, success: bool, execution_time: float):
        """Update conversion metrics"""
        if success:
            self.conversion_metrics['successful_conversions'] += 1
        else:
            self.conversion_metrics['failed_conversions'] += 1
            
        total_conversions = (self.conversion_metrics['successful_conversions'] + 
                            self.conversion_metrics['failed_conversions'])
        
        # Update average time
        current_avg = self.conversion_metrics['average_time']
        self.conversion_metrics['average_time'] = ((current_avg * (total_conversions - 1)) + 
                                                execution_time) / total_conversions
    
    def _add_to_history(self, etl_tool: str, target_database: str, success: bool):
        """Add conversion to history"""
        self._history.append({
            'timestamp': datetime.now().isoformat(),
            'etl_tool': etl_tool,
            'target_database': target_database,
            'success': success
        })
        
        # Keep history limited to last 100 entries
        if len(self._history) > 100:
            self._history = self._history[-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get conversion metrics"""
        return {
            'metrics': self.conversion_metrics,
            'history': self._history[-10:],  # Return last 10 entries
            'success_rate': (self.conversion_metrics['successful_conversions'] / 
                           max(1, self.conversion_metrics['successful_conversions'] + 
                               self.conversion_metrics['failed_conversions']))
        }

async def main():
    """Example usage with sample ETL code"""
    print("🔄 Initializing Advanced ETL-to-SQL Converter...")
    
    # Sample Informatica code for demonstration
    informatica_xml = """
    <MAPPING NAME="customer_sales_analysis">
        <SOURCE NAME="CUSTOMERS" DATABASETYPE="Oracle">
            <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="NUMBER"/>
            <TRANSFORMFIELD NAME="CUSTOMER_NAME" DATATYPE="VARCHAR2"/>
            <TRANSFORMFIELD NAME="REGION" DATATYPE="VARCHAR2"/>
        </SOURCE>
        <SOURCE NAME="SALES" DATABASETYPE="Oracle">
            <TRANSFORMFIELD NAME="SALE_ID" DATATYPE="NUMBER"/>
            <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="NUMBER"/>
            <TRANSFORMFIELD NAME="AMOUNT" DATATYPE="NUMBER"/>
            <TRANSFORMFIELD NAME="SALE_DATE" DATATYPE="DATE"/>
        </SOURCE>
        <TRANSFORMATION TYPE="Joiner" NAME="JNR_CUST_SALES">
            <CONDITION>CUSTOMERS.CUSTOMER_ID = SALES.CUSTOMER_ID</CONDITION>
        </TRANSFORMATION>
        <TRANSFORMATION TYPE="Aggregator" NAME="AGG_SALES_BY_REGION">
            <GROUP_BY>REGION</GROUP_BY>
            <EXPRESSION PORT="TOTAL_SALES">SUM(AMOUNT)</EXPRESSION>
            <EXPRESSION PORT="AVG_SALES">AVG(AMOUNT)</EXPRESSION>
        </TRANSFORMATION>
    </MAPPING>
    """
    
    # Sample schema info
    schema_info = {
        "customers": {
            "customer_id": "int primary key",
            "customer_name": "varchar(100)",
            "region": "varchar(50)"
        },
        "sales": {
            "sale_id": "int primary key",
            "customer_id": "int foreign key",
            "amount": "decimal(10,2)",
            "sale_date": "date"
        }
    }
    
    # Sample business rules
    business_rules = [
        "Only include active customers",
        "Sales amounts must be positive",
        "Group by region for reporting",
        "Calculate both total and average sales per region"
    ]
    
    try:
        # For real usage, replace with your API key
        # For demonstration, we'll use environment variables or placeholders
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI_GPT4,
            api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
            model_name="gpt-4-turbo",
            temperature=0.1
        )
        
        converter = AdvancedETLConverter(llm_config)
        
        print("🚀 Starting ETL-to-SQL conversion with data lineage and quality checks...")
        
        result = await converter.convert(
            etl_code=informatica_xml,
            etl_tool=ETLToolType.INFORMATICA,
            target_database=DatabaseType.SPARK_SQL,
            schema_info=schema_info,
            business_rules=business_rules
        )
        
        if result.success:
            print("✅ Conversion successful!")
            print("\n📊 Generated Spark SQL:")
            print(result.sql_code)
            print("\n📚 Documentation:")
            print(result.documentation[:500] + "..." if result.documentation and len(result.documentation) > 500 else result.documentation)
            print("\n⏱️ Execution time:", result.execution_time, "seconds")
        else:
            print(f"❌ Conversion failed: {result.error}")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
