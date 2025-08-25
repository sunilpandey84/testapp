#!/usr/bin/env python3
"""
LLM-Powered ETL to SQL Converter
Real implementation using Language Models for intelligent conversion
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
import ast
import re
from datetime import datetime

# For LLM integration - you'll need to install these
try:
    import openai  # pip install openai
    from anthropic import Anthropic  # pip install anthropic
except ImportError:
    print("Please install: pip install openai anthropic")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI_GPT4 = "openai_gpt4"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    AZURE_OPENAI = "azure_openai"

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 4000

@dataclass
class ConversionContext:
    etl_tool: str
    source_code: str
    target_database: str
    schema_info: Dict[str, Any]
    business_rules: List[str]
    metadata: Dict[str, Any]

class LLMOrchestrator:
    """Orchestrates LLM calls for ETL conversion"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider == LLMProvider.OPENAI_GPT4:
            return openai.OpenAI(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.ANTHROPIC_CLAUDE:
            return Anthropic(api_key=self.config.api_key)
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            return openai.AzureOpenAI(
                api_key=self.config.api_key,
                api_version="2023-12-01-preview",
                azure_endpoint=self.config.metadata.get('azure_endpoint')
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def convert_etl_to_sql(self, context: ConversionContext) -> Dict[str, Any]:
        """Main LLM-powered conversion function"""
        try:
            # Step 1: Analyze ETL code structure
            analysis_prompt = self._build_analysis_prompt(context)
            analysis_result = await self._call_llm(analysis_prompt, "analysis")
            
            # Step 2: Generate SQL with business context
            sql_prompt = self._build_sql_generation_prompt(context, analysis_result)
            sql_result = await self._call_llm(sql_prompt, "sql_generation")
            
            # Step 3: Optimize for target database
            optimization_prompt = self._build_optimization_prompt(sql_result, context.target_database)
            optimized_sql = await self._call_llm(optimization_prompt, "optimization")
            
            # Step 4: Generate documentation
            doc_prompt = self._build_documentation_prompt(context, optimized_sql)
            documentation = await self._call_llm(doc_prompt, "documentation")
            
            return {
                'success': True,
                'sql_code': optimized_sql,
                'documentation': documentation,
                'analysis': analysis_result,
                'conversion_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'llm_provider': self.config.provider.value,
                    'model_used': self.config.model_name,
                    'target_database': context.target_database
                }
            }
            
        except Exception as e:
            logger.error(f"LLM conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_code': None
            }

    def _build_analysis_prompt(self, context: ConversionContext) -> str:
        """Build prompt for ETL code analysis"""
        return f"""
You are an expert data engineer analyzing ETL code for conversion to SQL.

**Task**: Analyze the following {context.etl_tool} code and extract:
1. Data sources and their schemas
2. Transformation logic and business rules
3. Data quality checks and validations
4. Output targets and data flow
5. Complex logic that needs special SQL handling

**ETL Tool**: {context.etl_tool}
**Target Database**: {context.target_database}

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

Analyze thoroughly and be specific about SQL implementation strategies.
"""

    def _build_sql_generation_prompt(self, context: ConversionContext, analysis: str) -> str:
        """Build prompt for SQL code generation"""
        return f"""
You are an expert SQL developer generating optimized SQL code from ETL logic.

**Task**: Convert the analyzed ETL logic to equivalent SQL code for {context.target_database}.

**Analysis Results**:
{analysis}

**Original ETL Code**:
```
{context.source_code}
```

**Requirements**:
1. Generate functionally equivalent SQL code
2. Maintain all business rules and data transformations
3. Include proper error handling and data quality checks
4. Use {context.target_database}-specific SQL features and functions
5. Optimize for performance with proper indexing hints where applicable
6. Include comprehensive comments explaining the logic

**SQL Best Practices to Follow**:
- Use CTEs (Common Table Expressions) for complex transformations
- Implement proper NULL handling
- Use appropriate data types for {context.target_database}
- Include data quality validation
- Structure code for readability and maintainability
- Add performance optimization hints

**Output Format**:
Generate clean, production-ready SQL code with:
1. Header comments explaining the conversion
2. Well-structured CTEs for complex logic
3. Inline comments for business rules
4. Data quality checks
5. Final SELECT or INSERT/UPDATE statements

Focus on correctness, performance, and maintainability.
"""

    def _build_optimization_prompt(self, sql_code: str, target_db: str) -> str:
        """Build prompt for database-specific optimization"""
        return f"""
You are a database performance expert optimizing SQL for {target_db}.

**Task**: Optimize the following SQL code specifically for {target_db}, focusing on:

1. **Database-specific functions**: Use {target_db} native functions
2. **Performance optimization**: Query structure, indexing, partitioning
3. **Best practices**: {target_db} coding standards and conventions
4. **Resource efficiency**: Memory usage, I/O optimization
5. **Maintainability**: Code organization and documentation

**SQL Code to Optimize**:
```sql
{sql_code}
```

**{target_db} Specific Optimizations**:
- Replace generic functions with {target_db} equivalents
- Use {target_db} specific data types
- Apply {target_db} performance hints and optimizations
- Structure for {target_db} query planner
- Include {target_db} specific error handling

**Output**: 
Return the optimized SQL code with:
1. Performance improvements highlighted in comments
2. Database-specific functions used
3. Optimization rationale explained
4. Any indexing or partitioning recommendations

Ensure the code is production-ready for {target_db}.
"""

    def _build_documentation_prompt(self, context: ConversionContext, sql_code: str) -> str:
        """Build prompt for generating comprehensive documentation"""
        return f"""
You are a technical documentation expert creating comprehensive documentation for ETL-to-SQL conversion.

**Task**: Generate detailed documentation for the converted SQL code.

**Original ETL Tool**: {context.etl_tool}
**Target Database**: {context.target_database}

**Generated SQL Code**:
```sql
{sql_code}
```

**Documentation Requirements**:
1. **Executive Summary**: High-level overview of the conversion
2. **Technical Specifications**: Detailed technical details
3. **Business Logic Mapping**: How ETL logic maps to SQL
4. **Performance Considerations**: Expected performance characteristics
5. **Maintenance Guide**: How to maintain and modify the code
6. **Testing Recommendations**: How to validate the conversion
7. **Deployment Notes**: Deployment considerations

**Output Format**: Markdown documentation with clear sections and examples.

Generate comprehensive, professional documentation suitable for technical teams.
"""

    async def _call_llm(self, prompt: str, task_type: str) -> str:
        """Make actual LLM API call"""
        try:
            if self.config.provider == LLMProvider.ANTHROPIC_CLAUDE:
                response = await self._call_anthropic(prompt)
            elif self.config.provider == LLMProvider.OPENAI_GPT4:
                response = await self._call_openai(prompt)
            else:
                raise ValueError(f"LLM provider not implemented: {self.config.provider}")
            
            logger.info(f"LLM call successful for task: {task_type}")
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed for {task_type}: {e}")
            raise

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
                "content": "You are an expert data engineer specializing in ETL to SQL conversion."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

class ETLCodeAnalyzer:
    """Analyzes ETL code to extract context for LLM"""
    
    def analyze_informatica(self, xml_content: str) -> Dict[str, Any]:
        """Analyze Informatica mapping XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Extract detailed context
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
            
            # Extract transformation details
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
                
                transformations.append(trans_info)
            
            return {
                'sources': sources,
                'transformations': transformations,
                'complexity': self._assess_informatica_complexity(transformations)
            }
            
        except Exception as e:
            logger.error(f"Informatica analysis failed: {e}")
            return {}

    def analyze_python_etl(self, python_code: str) -> Dict[str, Any]:
        """Analyze Python ETL code using AST"""
        try:
            tree = ast.parse(python_code)
            
            functions = []
            imports = []
            operations = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'operations': self._extract_pandas_operations(node),
                        'complexity': 'medium'
                    }
                    functions.append(func_info)
                
                elif isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr'):
                        operations.append({
                            'operation': node.func.attr,
                            'context': ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                        })
            
            return {
                'imports': imports,
                'functions': functions,
                'operations': operations,
                'complexity': self._assess_python_complexity(functions)
            }
            
        except Exception as e:
            logger.error(f"Python analysis failed: {e}")
            return {}

    def _extract_pandas_operations(self, func_node) -> List[Dict]:
        """Extract pandas operations from function"""
        operations = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                op_name = node.func.attr
                if op_name in ['groupby', 'merge', 'join', 'agg', 'apply', 'transform']:
                    operations.append({
                        'type': op_name,
                        'complexity': 'high' if op_name in ['apply', 'transform'] else 'medium'
                    })
        return operations

    def _assess_informatica_complexity(self, transformations: List[Dict]) -> str:
        """Assess complexity of Informatica mapping"""
        complex_types = ['Lookup', 'Aggregator', 'Joiner', 'Union', 'Rank']
        complex_count = sum(1 for t in transformations if t['type'] in complex_types)
        
        if complex_count > 3:
            return 'high'
        elif complex_count > 1:
            return 'medium'
        else:
            return 'low'

    def _assess_python_complexity(self, functions: List[Dict]) -> str:
        """Assess complexity of Python ETL code"""
        total_ops = sum(len(f['operations']) for f in functions)
        if total_ops > 10:
            return 'high'
        elif total_ops > 5:
            return 'medium'
        else:
            return 'low'

class IntelligentETLConverter:
    """Main LLM-powered ETL converter"""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_orchestrator = LLMOrchestrator(llm_config)
        self.code_analyzer = ETLCodeAnalyzer()

    async def convert(self, 
                     etl_code: str,
                     etl_tool: str,
                     target_database: str,
                     schema_info: Optional[Dict] = None,
                     business_rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main conversion method using LLM intelligence"""
        
        try:
            # Step 1: Analyze ETL code structure
            if etl_tool.lower() == 'informatica':
                analysis = self.code_analyzer.analyze_informatica(etl_code)
            elif etl_tool.lower() in ['python', 'pandas']:
                analysis = self.code_analyzer.analyze_python_etl(etl_code)
            else:
                analysis = {}  # LLM will handle analysis
            
            # Step 2: Build conversion context
            context = ConversionContext(
                etl_tool=etl_tool,
                source_code=etl_code,
                target_database=target_database,
                schema_info=schema_info or {},
                business_rules=business_rules or [],
                metadata={'analysis': analysis}
            )
            
            # Step 3: Let LLM do the intelligent conversion
            result = await self.llm_orchestrator.convert_etl_to_sql(context)
            
            # Step 4: Add analysis metadata
            if result['success']:
                result['code_analysis'] = analysis
                result['conversion_context'] = {
                    'etl_tool': etl_tool,
                    'target_database': target_database,
                    'complexity': analysis.get('complexity', 'unknown')
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_code': None
            }

# Example usage with real LLM integration
async def main():
    """Example usage of LLM-powered ETL converter"""
    
    # Configure your LLM (you'll need actual API keys)
    llm_config = LLMConfig(
        provider=LLMProvider.ANTHROPIC_CLAUDE,
        api_key="your-anthropic-api-key",  # Replace with actual key
        model_name="claude-3-sonnet-20240229",
        temperature=0.1,
        max_tokens=4000
    )
    
    converter = IntelligentETLConverter(llm_config)
    
    # Example 1: Complex Informatica mapping
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
    
    business_rules = [
        "Only include active customers",
        "Sales amounts must be positive",
        "Group by region for reporting",
        "Calculate both total and average sales per region"
    ]
    
    print("🚀 Starting LLM-powered ETL conversion...")
    
    result = await converter.convert(
        etl_code=informatica_xml,
        etl_tool="informatica",
        target_database="postgresql",
        schema_info=schema_info,
        business_rules=business_rules
    )
    
    if result['success']:
        print("✅ Conversion successful!")
        print("\n📊 Generated SQL:")
        print(result['sql_code'])
        print("\n📚 Documentation:")
        print(result['documentation'])
        print(f"\n🔍 Complexity: {result.get('conversion_context', {}).get('complexity', 'unknown')}")
    else:
        print(f"❌ Conversion failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())