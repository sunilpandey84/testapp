#!/usr/bin/env python3
"""
ETL to SQL Converter - Working Prototype
A comprehensive solution for converting ETL logic from various tools to SQL
"""

import json
import re
import ast
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLTool(Enum):
    INFORMATICA = "informatica"
    PYTHON_PANDAS = "python_pandas"
    PYSPARK = "pyspark"
    TALEND = "talend"
    KETTLE_PDI = "kettle_pdi"

@dataclass
class TransformationRule:
    rule_type: str
    source_pattern: str
    sql_template: str
    description: str

@dataclass
class ETLContext:
    source_tables: List[Dict[str, Any]]
    target_tables: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]
    business_rules: List[str]
    schema_info: Dict[str, Any]

class PatternLibrary:
    """Library of common ETL patterns and their SQL equivalents"""
    
    def __init__(self):
        self.patterns = {
            # Informatica patterns
            'informatica': [
                TransformationRule(
                    rule_type="null_handling",
                    source_pattern=r"IIF\(ISNULL\((\w+)\),\s*([^,]+),\s*\1\)",
                    sql_template="COALESCE({0}, {1})",
                    description="Convert Informatica IIF ISNULL to COALESCE"
                ),
                TransformationRule(
                    rule_type="string_concat",
                    source_pattern=r"(\w+)\s*\|\|\s*'([^']+)'\s*\|\|\s*(\w+)",
                    sql_template="{0} || '{1}' || {2}",
                    description="String concatenation"
                ),
                TransformationRule(
                    rule_type="date_function",
                    source_pattern=r"SYSDATE",
                    sql_template="CURRENT_TIMESTAMP",
                    description="Convert SYSDATE to standard SQL"
                )
            ],
            # Python pandas patterns
            'python_pandas': [
                TransformationRule(
                    rule_type="groupby_agg",
                    source_pattern=r"\.groupby\('(\w+)'\)\.agg\(\{'(\w+)':\s*'(\w+)'\}\)",
                    sql_template="SELECT {0}, {2}({1}) FROM table GROUP BY {0}",
                    description="Convert pandas groupby aggregation"
                ),
                TransformationRule(
                    rule_type="filter",
                    source_pattern=r"\[df\['(\w+)'\]\s*([><=!]+)\s*([^]]+)\]",
                    sql_template="WHERE {0} {1} {2}",
                    description="Convert pandas filtering"
                )
            ]
        }

class InformaticaParser:
    """Parser for Informatica PowerCenter mappings"""
    
    def parse_mapping_xml(self, xml_content: str) -> ETLContext:
        """Parse Informatica mapping XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Extract source definitions
            sources = []
            for source in root.findall(".//SOURCE"):
                sources.append({
                    'name': source.get('NAME', ''),
                    'type': source.get('DATABASETYPE', 'Unknown'),
                    'columns': self._extract_columns(source)
                })
            
            # Extract transformations
            transformations = []
            for trans in root.findall(".//TRANSFORMATION"):
                trans_data = {
                    'type': trans.get('TYPE', ''),
                    'name': trans.get('NAME', ''),
                    'expressions': self._extract_expressions(trans)
                }
                transformations.append(trans_data)
            
            # Extract targets
            targets = []
            for target in root.findall(".//TARGET"):
                targets.append({
                    'name': target.get('NAME', ''),
                    'type': target.get('DATABASETYPE', 'Unknown'),
                    'columns': self._extract_columns(target)
                })
            
            return ETLContext(
                source_tables=sources,
                target_tables=targets,
                transformations=transformations,
                business_rules=[],
                schema_info={}
            )
        
        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {e}")
            raise
    
    def _extract_columns(self, element):
        """Extract column information from XML element"""
        columns = []
        for col in element.findall(".//TRANSFORMFIELD"):
            columns.append({
                'name': col.get('NAME', ''),
                'datatype': col.get('DATATYPE', ''),
                'precision': col.get('PRECISION', ''),
                'scale': col.get('SCALE', '')
            })
        return columns
    
    def _extract_expressions(self, transformation):
        """Extract expressions from transformation"""
        expressions = []
        for expr in transformation.findall(".//EXPRESSION"):
            expressions.append({
                'port': expr.get('PORT', ''),
                'expression': expr.text or ''
            })
        return expressions

class PythonETLParser:
    """Parser for Python-based ETL scripts"""
    
    def parse_python_script(self, script_content: str) -> ETLContext:
        """Parse Python ETL script using AST"""
        try:
            tree = ast.parse(script_content)
            
            transformations = []
            business_rules = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    transformations.append(func_info)
                
                elif isinstance(node, ast.Assign):
                    assignment_info = self._analyze_assignment(node)
                    if assignment_info:
                        business_rules.append(assignment_info)
            
            return ETLContext(
                source_tables=[],  # Would be extracted from pandas.read_* calls
                target_tables=[],   # Would be extracted from df.to_* calls
                transformations=transformations,
                business_rules=business_rules,
                schema_info={}
            )
        
        except SyntaxError as e:
            logger.error(f"Error parsing Python script: {e}")
            raise
    
    def _analyze_function(self, func_node):
        """Analyze function definition"""
        return {
            'type': 'function',
            'name': func_node.name,
            'operations': self._extract_operations(func_node)
        }
    
    def _analyze_assignment(self, assign_node):
        """Analyze assignment statement"""
        if isinstance(assign_node.value, ast.BinOp):
            return f"Binary operation: {ast.dump(assign_node)}"
        return None
    
    def _extract_operations(self, func_node):
        """Extract operations from function body"""
        operations = []
        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                operations.append({
                    'type': 'assignment',
                    'target': ast.dump(stmt.targets[0]),
                    'value': ast.dump(stmt.value)
                })
        return operations

class SQLGenerator:
    """Generate SQL from parsed ETL context using patterns"""
    
    def __init__(self, pattern_library: PatternLibrary):
        self.patterns = pattern_library
        self.sql_templates = {
            'select': "SELECT {columns} FROM {tables}",
            'insert': "INSERT INTO {table} ({columns}) VALUES ({values})",
            'update': "UPDATE {table} SET {assignments} WHERE {conditions}",
            'with_cte': "WITH {cte_name} AS ({cte_query}) {main_query}"
        }
    
    def generate_sql(self, context: ETLContext, tool_type: ETLTool) -> str:
        """Generate SQL from ETL context"""
        try:
            if tool_type == ETLTool.INFORMATICA:
                return self._generate_informatica_sql(context)
            elif tool_type == ETLTool.PYTHON_PANDAS:
                return self._generate_pandas_sql(context)
            else:
                raise ValueError(f"Unsupported tool type: {tool_type}")
        
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    def _generate_informatica_sql(self, context: ETLContext) -> str:
        """Generate SQL for Informatica mappings"""
        sql_parts = []
        
        # Process transformations
        select_columns = []
        from_tables = []
        where_conditions = []
        
        for source in context.source_tables:
            from_tables.append(source['name'])
        
        for transformation in context.transformations:
            if transformation['type'] == 'Expression':
                for expr in transformation['expressions']:
                    # Apply pattern matching
                    converted_expr = self._apply_patterns(
                        expr['expression'], 
                        'informatica'
                    )
                    select_columns.append(f"{converted_expr} as {expr['port']}")
            
            elif transformation['type'] == 'Filter':
                # Handle filter transformations
                for expr in transformation['expressions']:
                    if 'WHERE' in expr['expression'].upper():
                        condition = self._apply_patterns(
                            expr['expression'], 
                            'informatica'
                        )
                        where_conditions.append(condition)
        
        # Build final SQL
        if not select_columns:
            select_columns = ['*']
        
        sql = f"SELECT {', '.join(select_columns)}"
        sql += f" FROM {', '.join(from_tables)}"
        
        if where_conditions:
            sql += f" WHERE {' AND '.join(where_conditions)}"
        
        # Add target insert if available
        if context.target_tables:
            target = context.target_tables[0]
            insert_sql = f"INSERT INTO {target['name']} ({', '.join([col['name'] for col in target['columns']])}) "
            sql = insert_sql + "(" + sql + ")"
        
        return sql
    
    def _generate_pandas_sql(self, context: ETLContext) -> str:
        """Generate SQL for pandas operations"""
        sql_parts = []
        
        for transformation in context.transformations:
            if transformation['type'] == 'function':
                operations = transformation['operations']
                
                # Analyze operations to build SQL
                for op in operations:
                    if 'groupby' in op['value']:
                        # Convert groupby to SQL
                        sql_parts.append("-- Groupby operation converted")
                    elif 'filter' in op['value']:
                        # Convert filter to WHERE clause
                        sql_parts.append("-- Filter operation converted")
        
        return "-- Generated SQL from pandas operations\n" + "\n".join(sql_parts)
    
    def _apply_patterns(self, expression: str, tool_type: str) -> str:
        """Apply pattern matching to convert expressions"""
        converted = expression
        
        if tool_type in self.patterns.patterns:
            for pattern in self.patterns.patterns[tool_type]:
                match = re.search(pattern.source_pattern, expression)
                if match:
                    groups = match.groups()
                    converted = pattern.sql_template.format(*groups)
                    logger.info(f"Applied pattern: {pattern.description}")
                    break
        
        return converted

class ETLToSQLConverter:
    """Main converter class"""
    
    def __init__(self):
        self.pattern_library = PatternLibrary()
        self.parsers = {
            ETLTool.INFORMATICA: InformaticaParser(),
            ETLTool.PYTHON_PANDAS: PythonETLParser(),
        }
        self.sql_generator = SQLGenerator(self.pattern_library)
    
    def convert(self, 
                input_content: str, 
                tool_type: ETLTool, 
                target_db: str = "postgresql") -> Dict[str, Any]:
        """Convert ETL logic to SQL"""
        
        try:
            # Parse input based on tool type
            if tool_type not in self.parsers:
                raise ValueError(f"Unsupported tool type: {tool_type}")
            
            parser = self.parsers[tool_type]
            
            if tool_type == ETLTool.INFORMATICA:
                context = parser.parse_mapping_xml(input_content)
            elif tool_type == ETLTool.PYTHON_PANDAS:
                context = parser.parse_python_script(input_content)
            
            # Generate SQL
            sql_code = self.sql_generator.generate_sql(context, tool_type)
            
            # Optimize for target database
            optimized_sql = self._optimize_for_database(sql_code, target_db)
            
            return {
                'success': True,
                'sql_code': optimized_sql,
                'context': context,
                'metadata': {
                    'source_tool': tool_type.value,
                    'target_db': target_db,
                    'transformations_count': len(context.transformations)
                }
            }
        
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_code': None
            }
    
    def _optimize_for_database(self, sql: str, target_db: str) -> str:
        """Optimize SQL for specific database"""
        optimizations = {
            'postgresql': {
                'SYSDATE': 'CURRENT_TIMESTAMP',
                'NVL(': 'COALESCE(',
                'DECODE(': 'CASE WHEN'
            },
            'mysql': {
                'CURRENT_TIMESTAMP': 'NOW()',
                'COALESCE(': 'IFNULL('
            }
        }
        
        if target_db in optimizations:
            for old, new in optimizations[target_db].items():
                sql = sql.replace(old, new)
        
        return sql

# Example usage and testing
def main():
    """Example usage of the ETL converter"""
    
    converter = ETLToSQLConverter()
    
    # Example 1: Informatica XML conversion
    informatica_xml = """
    <MAPPING NAME="m_customer_load">
        <SOURCE NAME="CUSTOMER_SRC" DATABASETYPE="Oracle">
            <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="number"/>
            <TRANSFORMFIELD NAME="FIRST_NAME" DATATYPE="varchar2"/>
            <TRANSFORMFIELD NAME="LAST_NAME" DATATYPE="varchar2"/>
        </SOURCE>
        <TRANSFORMATION TYPE="Expression" NAME="EXP_CUSTOMER">
            <EXPRESSION PORT="FULL_NAME">FIRST_NAME || ' ' || LAST_NAME</EXPRESSION>
            <EXPRESSION PORT="CUSTOMER_KEY">IIF(ISNULL(CUSTOMER_ID), 0, CUSTOMER_ID)</EXPRESSION>
        </TRANSFORMATION>
        <TARGET NAME="CUSTOMER_TGT" DATABASETYPE="PostgreSQL">
            <TRANSFORMFIELD NAME="CUSTOMER_KEY" DATATYPE="integer"/>
            <TRANSFORMFIELD NAME="FULL_NAME" DATATYPE="varchar"/>
        </TARGET>
    </MAPPING>
    """
    
    print("Converting Informatica mapping to SQL...")
    result = converter.convert(informatica_xml, ETLTool.INFORMATICA, "postgresql")
    
    if result['success']:
        print("Conversion successful!")
        print("Generated SQL:")
        print(result['sql_code'])
        print("\nMetadata:")
        print(json.dumps(result['metadata'], indent=2))
    else:
        print(f"Conversion failed: {result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Python pandas conversion
    python_script = """
def process_sales_data(df):
    # Calculate total amount
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    # Filter high-value transactions
    df_filtered = df[df['total_amount'] > 100]
    
    # Group by customer
    result = df_filtered.groupby('customer_id').agg({
        'total_amount': 'sum',
        'order_date': 'max'
    })
    
    return result
    """
    
    print("Converting Python ETL script to SQL...")
    result = converter.convert(python_script, ETLTool.PYTHON_PANDAS, "postgresql")
    
    if result['success']:
        print("Conversion successful!")
        print("Generated SQL:")
        print(result['sql_code'])
    else:
        print(f"Conversion failed: {result['error']}")

if __name__ == "__main__":
    main()