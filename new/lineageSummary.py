import json
import asyncio
import traceback
import os
from typing import Dict, List, Any, Optional, Set, Tuple, Annotated
from dataclasses import dataclass
from enum import Enum
import sqlite3
from datetime import datetime
import logging
from collections import Counter

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

COORDINATOR_SYSTEM_PROMPT = """You are an intelligent Data Lineage Coordinator. Your role is to:

1. Analyze user requests for data lineage tracing
2. Determine the most appropriate lineage strategy (contract-based, element-based, or contract-summarization)
3. Extract key parameters from natural language requests
4. Handle ambiguous requests by asking clarifying questions
5. Route the request to the appropriate specialized agent

Guidelines:
- If user mentions "summarize", "summary", "analyze contract", "quality issues", "contract analysis" -> choose contract_summarization
- If user mentions a specific contract name, pipeline, or data flow -> choose contract_based lineage
- If user mentions specific data elements, fields, or columns -> choose element_based lineage  
- If the request is ambiguous, ask for clarification
- Always extract the traversal direction (upstream, downstream, or bidirectional) from context
- Be conversational and helpful in your responses

Current request analysis: Determine the lineage type, extract parameters, and decide next steps.
"""

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert Data Contract Summarization Agent. Your responsibilities:

1. Analyze data contracts comprehensively for quality and completeness
2. Identify metadata quality issues and missing information
3. Provide statistical analysis of pipeline associations
4. Analyze source and target system distributions
5. Generate actionable recommendations for contract improvement
6. Assess data governance maturity and compliance

Quality Assessment Guidelines:
- Check for missing or incomplete contract descriptions
- Identify contracts without proper ownership information
- Flag contracts with insufficient system information
- Analyze pipeline distribution and complexity
- Assess transformation coverage and documentation
- Identify potential data quality bottlenecks

Statistical Analysis Focus:
- Count of pipelines per contract
- Distribution of source and target systems
- Source and target type analysis
- Transformation complexity metrics
- Data flow patterns and dependencies

Available tools: query_contract_by_name, query_pipelines_by_contract, analyze_contract_quality, 
get_contract_statistics, analyze_pipeline_distribution, get_system_analysis
"""

CONTRACT_SYSTEM_PROMPT = """You are an expert Contract Analysis Agent for data lineage. Your responsibilities:

1. Analyze data contracts and their associated pipelines
2. Identify all relevant ETL processes within a contract
3. Map dependencies between pipelines
4. Assess the complexity and scope of lineage tracing
5. Provide intelligent insights about the data flow

Guidelines:
- Thoroughly examine contract details, descriptions, and ownership
- Identify potential data quality issues or bottlenecks
- Suggest optimization opportunities
- Flag any missing or incomplete information
- Provide context about the business impact of the lineage

Available tools: query_contract_by_name, query_pipelines_by_contract, query_pipeline_dependencies, query_element_mappings_by_queries
"""

ELEMENT_SYSTEM_PROMPT = """You are an intelligent Element Tracing Agent. Your role is to:

1. Trace data element connections across the entire data ecosystem
2. Understand transformation logic and business rules
3. Identify data quality and governance issues
4. Provide insights about data flow patterns
5. Detect circular dependencies or anomalies

Guidelines:
- Analyze transformation rules and their business meaning
- Consider data quality implications of each transformation
- Identify potential data lineage gaps or missing connections
- Suggest data governance improvements
- Be thorough in tracing all relevant connections

Available tools: find_element_by_name, trace_element_connections, query_element_mappings_by_queries
"""

CONSTRUCTION_SYSTEM_PROMPT = """You are a Lineage Graph Construction Expert. Your responsibilities:

1. Build comprehensive and accurate lineage graphs
2. Optimize graph structure for clarity and usability
3. Add meaningful metadata and context
4. Identify patterns and insights in the lineage
5. Assess complexity and recommend visualization strategies

Guidelines:
- Create clear, hierarchical graph structures
- Add rich metadata to nodes and edges
- Identify critical data flow paths
- Highlight transformation complexity
- Suggest graph simplification for better understanding
- Flag potential data governance issues
"""

APPROVAL_SYSTEM_PROMPT = """You are a Human Interaction Specialist for data lineage workflows. Your role is to:

1. Process human feedback intelligently
2. Handle ambiguous responses and requests for clarification
3. Validate user selections and inputs
4. Provide helpful guidance and explanations
5. Ensure smooth human-AI collaboration

Guidelines:
- Be patient and understanding with human responses
- Provide clear options and explanations
- Handle errors gracefully with helpful suggestions
- Validate inputs before processing
- Ask follow-up questions when needed
"""


class LineageState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    lineage_type: str
    input_parameter: str
    contract_name: Optional[str]
    element_name: Optional[str]
    traversal_direction: str
    current_step: str
    query_results: Dict[str, Any]
    lineage_nodes: List[Dict[str, Any]]
    lineage_edges: List[Dict[str, Any]]

    # Fields for recursive element tracing
    element_queue: List[str]
    traced_elements: Set[str]

    # Fields for Human-in-the-Loop (HITL)
    requires_human_approval: bool
    human_approval_message: Optional[str]
    human_feedback: Optional[Dict[str, Any]]

    # Enhanced fields for LLM responses
    llm_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    complexity_score: Optional[int]

    # New fields for contract summarization
    contract_summary: Optional[Dict[str, Any]]
    quality_assessment: Optional[Dict[str, Any]]
    statistical_analysis: Optional[Dict[str, Any]]

    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]


class LineageRequest(BaseModel):
    """Enhanced request model to handle natural language inputs"""
    query: str = Field(description="Natural language query for lineage tracing")
    context: Optional[str] = Field(None, description="Additional context about the request")
    preferred_output: Optional[str] = Field(None, description="Preferred output format (graph, table, summary)")
    max_depth: Optional[int] = Field(5, description="Maximum depth for lineage tracing")


# --- Enums and Dataclasses ---
class LineageType(Enum):
    CONTRACT_BASED = "contract_based"
    ELEMENT_BASED = "element_based"
    CONTRACT_SUMMARIZATION = "contract_summarization"


class TraversalDirection(Enum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BIDIRECTIONAL = "bidirectional"


# --- Database Management ---
class DatabaseManager:
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = "../metadata.db"
        if not os.path.exists(db_path):
            logger.info("Database not found. Initializing new database...")
            self.init_database()

    def init_database(self):
        """Initialize the database with the required schema and sample metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
                             CREATE TABLE IF NOT EXISTS data_contracts
                             (
                                 v_contract_code
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 v_contract_name
                                 TEXT,
                                 v_contract_description
                                 TEXT,
                                 v_source_owner
                                 TEXT,
                                 v_ingestion_owner
                                 TEXT,
                                 v_source_system
                                 TEXT,
                                 v_target_system
                                 TEXT
                             );
                             CREATE TABLE IF NOT EXISTS etl_pipeline_metadata
                             (
                                 v_query_code
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 v_query_description
                                 TEXT,
                                 v_target_table_or_object
                                 TEXT,
                                 v_source_table_or_object
                                 TEXT,
                                 v_source_type
                                 TEXT,
                                 v_target_type
                                 TEXT,
                                 v_from_clause
                                 TEXT,
                                 v_where_clause
                                 TEXT,
                                 v_contract_code
                                 TEXT,
                                 FOREIGN
                                 KEY
                             (
                                 v_contract_code
                             ) REFERENCES data_contracts
                             (
                                 v_contract_code
                             ));
                             CREATE TABLE IF NOT EXISTS etl_pipeline_dependency
                             (
                                 v_query_code
                                 TEXT,
                                 v_depends_on
                                 TEXT,
                                 FOREIGN
                                 KEY
                             (
                                 v_query_code
                             ) REFERENCES etl_pipeline_metadata
                             (
                                 v_query_code
                             ), FOREIGN KEY
                             (
                                 v_depends_on
                             ) REFERENCES etl_pipeline_metadata
                             (
                                 v_query_code
                             ));
                             CREATE TABLE IF NOT EXISTS business_dictionary
                             (
                                 v_business_element_code
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 v_business_definition
                                 TEXT
                             );
                             CREATE TABLE IF NOT EXISTS business_element_mapping
                             (
                                 v_data_element_code
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 v_data_element_name
                                 TEXT,
                                 v_table_name
                                 TEXT,
                                 v_business_element_code
                                 TEXT,
                                 FOREIGN
                                 KEY
                             (
                                 v_business_element_code
                             ) REFERENCES business_dictionary
                             (
                                 v_business_element_code
                             ));
                             CREATE TABLE IF NOT EXISTS transformation_rules
                             (
                                 v_transformation_code
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 v_transformation_rules
                                 TEXT
                             );
                             CREATE TABLE IF NOT EXISTS etl_element_mapping
                             (
                                 v_query_code
                                 TEXT,
                                 v_source_data_element_code
                                 TEXT,
                                 v_target_data_element_code
                                 TEXT,
                                 v_transformation_code
                                 TEXT,
                                 FOREIGN
                                 KEY
                             (
                                 v_query_code
                             ) REFERENCES etl_pipeline_metadata_metadata_metadata_metadata_metadata_metadata_metadata_metadata_metadata
                             (
                                 v_query_code
                             ), FOREIGN KEY
                             (
                                 v_source_data_element_code
                             ) REFERENCES business_element_mapping
                             (
                                 v_data_element_code
                             ), FOREIGN KEY
                             (
                                 v_target_data_element_code
                             ) REFERENCES business_element_mapping
                             (
                                 v_data_element_code
                             ), FOREIGN KEY
                             (
                                 v_transformation_code
                             ) REFERENCES transformation_rules
                             (
                                 v_transformation_code
                             ));
                             """)
        self._insert_sample_data(cursor)
        conn.commit()
        conn.close()

    def _insert_sample_data(self, cursor):
        """Insert comprehensive sample metadata."""
        cursor.executemany("INSERT OR REPLACE INTO business_dictionary VALUES (?, ?)",
                           [('BE001', 'Customer unique identifier'), ('BE002', 'Customer full name'),
                            ('BE003', 'Order monetary amount'), ('BE004', 'Order transaction date'),
                            ('BE005', 'Product unique identifier'), ('BE006', 'Product display name'),
                            ('BE007', 'Customer address information'), ('BE008', 'Aggregated sales metrics')])
        cursor.executemany("INSERT OR REPLACE INTO business_element_mapping VALUES (?, ?, ?, ?)",
                           [('DE001', 'customer_id', 'customers', 'BE001'),
                            ('DE002', 'customer_name', 'customers', 'BE002'),
                            ('DE003', 'customer_address', 'customers', 'BE007'),
                            ('DE004', 'order_amount', 'orders', 'BE003'), ('DE005', 'order_date', 'orders', 'BE004'),
                            ('DE006', 'product_id', 'products', 'BE005'),
                            ('DE007', 'product_name', 'products', 'BE006'),
                            ('DE008', 'cust_id', 'dim_customer', 'BE001'),
                            ('DE009', 'cust_name', 'dim_customer', 'BE002'),
                            ('DE010', 'cust_addr', 'dim_customer', 'BE007'),
                            ('DE011', 'total_amount', 'fact_orders', 'BE003'),
                            ('DE012', 'order_dt', 'fact_orders', 'BE004'), ('DE013', 'prod_id', 'fact_orders', 'BE005'),
                            ('DE014', 'sales_summary', 'agg_sales', 'BE008')])
        cursor.executemany("INSERT OR REPLACE INTO transformation_rules VALUES (?, ?)",
                           [('T001', 'DIRECT_COPY: Direct field mapping without transformation'),
                            ('T002', 'UPPER_CASE: Convert text to uppercase'),
                            ('T003', 'SUM_AGGREGATION: Sum aggregation across groups'),
                            ('T004', 'DATE_FORMAT_CONVERSION: Convert date format from YYYY-MM-DD to DD/MM/YYYY'),
                            ('T005', 'CONCATENATION: Combine multiple fields with separator'),
                            ('T006', 'LOOKUP_TRANSFORMATION: Foreign key lookup and replacement')])
        cursor.executemany("INSERT OR REPLACE INTO data_contracts VALUES (?, ?, ?, ?, ?, ?, ?)",
                           [('C001', 'Customer Data Pipeline',
                             'End-to-end customer data processing from CRM to warehouse', 'DataTeam', 'ETLTeam',
                             'CRM_System', 'DataWarehouse'),
                            ('C002', 'Order Processing Pipeline', 'Order data transformation and fact table creation',
                             'OrderTeam', 'ETLTeam', 'OrderSystem', 'DataWarehouse'),
                            ('C003', 'Product Analytics Pipeline', 'Product data enrichment and analytics preparation',
                             'ProductTeam', 'ETLTeam', 'ProductDB', 'AnalyticsDB'),
                            ('C004', 'Incomplete Contract', '', '', '', 'UnknownSystem', ''),
                            ('C005', 'Legacy System Migration', 'Migration of legacy data systems', 'LegacyTeam',
                             'MigrationTeam', 'MainframeSystem', 'CloudDataWarehouse')])
        cursor.executemany("INSERT OR REPLACE INTO etl_pipeline_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [('Q001', 'Load customer dimension table', 'dim_customer', 'customers', 'table', 'table',
                             'FROM customers c', 'WHERE c.active = 1 AND c.created_date >= CURRENT_DATE - 90', 'C001'),
                            ('Q002', 'Load order facts', 'fact_orders', 'orders o JOIN customers c', 'table', 'table',
                             'FROM orders o JOIN customers c ON o.customer_id = c.customer_id',
                             'WHERE o.order_date >= CURRENT_DATE - 30', 'C002'),
                            ('Q003', 'Aggregate sales data', 'agg_sales', 'fact_orders', 'table', 'table',
                             'FROM fact_orders fo', 'GROUP BY fo.prod_id, DATE_TRUNC(month, fo.order_dt)', 'C002'),
                            ('Q004', 'Product dimension load', 'dim_product', 'products', 'table', 'table',
                             'FROM products p', 'WHERE p.is_active = 1', 'C003'),
                            ('Q005', 'Sales analysis prep', 'sales_mart', 'fact_orders fo JOIN dim_product dp', 'view',
                             'table',
                             'FROM fact_orders fo JOIN dim_product dp ON fo.prod_id = dp.prod_id', '', 'C003'),
                            ('Q006', 'Legacy data extract', 'staging_legacy', 'legacy_tables', 'file', 'table',
                             'FROM legacy_system', '', 'C004'),
                            ('Q007', 'Customer profile build', 'customer_360', 'dim_customer dc JOIN fact_orders fo',
                             'table', 'view',
                             'FROM dim_customer dc JOIN fact_orders fo ON dc.cust_id = fo.cust_id', '', 'C005')])
        cursor.executemany("INSERT OR REPLACE INTO etl_pipeline_dependency VALUES (?, ?)",
                           [('Q002', 'Q001'), ('Q003', 'Q002'), ('Q005', 'Q003'), ('Q005', 'Q004'), ('Q007', 'Q001')])
        cursor.executemany("INSERT OR REPLACE INTO etl_element_mapping VALUES (?, ?, ?, ?)",
                           [('Q001', 'DE001', 'DE008', 'T001'), ('Q001', 'DE002', 'DE009', 'T002'),
                            ('Q001', 'DE003', 'DE010', 'T005'), ('Q002', 'DE004', 'DE011', 'T001'),
                            ('Q002', 'DE005', 'DE012', 'T004'), ('Q002', 'DE006', 'DE013', 'T001'),
                            ('Q003', 'DE011', 'DE014', 'T003')])

    def get_connection(self):
        return sqlite3.connect(self.db_path)


# DatabaseManager instance
db_manager_global = DatabaseManager()


@tool
def query_contract_by_name(contract_name: str) -> Dict[str, Any]:
    """Queries data contract table by contract name."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    contract_query_param = contract_name.replace(" ", "%")
    cursor.execute(
        "SELECT v_contract_code, v_contract_name, v_contract_description, v_source_owner, v_ingestion_owner, v_source_system, v_target_system FROM data_contracts WHERE v_contract_name LIKE ?",
        (f"%{contract_query_param}%",))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            "success": True,
            "contract_code": result[0],
            "contract_name": result[1],
            "description": result[2],
            "source_owner": result[3],
            "ingestion_owner": result[4],
            "source_system": result[5],
            "target_system": result[6]
        }
    return {"success": False, "error": f"Contract '{contract_name}' not found."}


@tool
def query_pipelines_by_contract(contract_code: str) -> Dict[str, Any]:
    """Gets all ETL pipelines for a contract code with enhanced details."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT v_query_code,
                          v_query_description,
                          v_target_table_or_object,
                          v_source_table_or_object,
                          v_source_type,
                          v_target_type,
                          v_from_clause,
                          v_where_clause
                   FROM etl_pipeline_metadata
                   WHERE v_contract_code = ?
                   """, (contract_code,))
    results = cursor.fetchall()
    conn.close()
    pipelines = [{
        "query_code": row[0],
        "description": row[1],
        "target_table": row[2],
        "source_table": row[3],
        "source_type": row[4],
        "target_type": row[5],
        "from_clause": row[6],
        "where_clause": row[7]
    } for row in results]
    return {"success": True, "pipelines": pipelines}


@tool
def analyze_contract_quality(contract_code: str) -> Dict[str, Any]:
    """Analyzes data contract quality and identifies metadata issues."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    # Get contract details
    cursor.execute("""
                   SELECT v_contract_code,
                          v_contract_name,
                          v_contract_description,
                          v_source_owner,
                          v_ingestion_owner,
                          v_source_system,
                          v_target_system
                   FROM data_contracts
                   WHERE v_contract_code = ?
                   """, (contract_code,))
    contract = cursor.fetchone()

    if not contract:
        conn.close()
        return {"success": False, "error": f"Contract {contract_code} not found"}

    quality_issues = []
    quality_score = 100

    # Check for missing or empty fields
    fields = {
        "Contract Name": contract[1],
        "Description": contract[2],
        "Source Owner": contract[3],
        "Ingestion Owner": contract[4],
        "Source System": contract[5],
        "Target System": contract[6]
    }

    for field_name, field_value in fields.items():
        if not field_value or field_value.strip() == "":
            quality_issues.append(f"Missing {field_name}")
            quality_score -= 15
        elif len(field_value.strip()) < 5:
            quality_issues.append(f"Insufficient {field_name} information (too short)")
            quality_score -= 10

    # Check for generic or placeholder values
    generic_values = ["unknown", "tbd", "todo", "placeholder", "temp", "test"]
    for field_name, field_value in fields.items():
        if field_value and any(generic in field_value.lower() for generic in generic_values):
            quality_issues.append(f"{field_name} contains generic/placeholder value")
            quality_score -= 12

    # Get pipeline count and check for documentation
    cursor.execute("SELECT COUNT(*) FROM etl_pipeline_metadata WHERE v_contract_code = ?", (contract_code,))
    pipeline_count = cursor.fetchone()[0]

    if pipeline_count == 0:
        quality_issues.append("No pipelines associated with this contract")
        quality_score -= 25

    # Check for pipeline descriptions
    cursor.execute("""
                   SELECT COUNT(*)
                   FROM etl_pipeline_metadata
                   WHERE v_contract_code = ?
                     AND (v_query_description IS NULL OR v_query_description = '')
                   """, (contract_code,))
    undocumented_pipelines = cursor.fetchone()[0]

    if undocumented_pipelines > 0:
        quality_issues.append(f"{undocumented_pipelines} pipelines lack proper descriptions")
        quality_score -= (undocumented_pipelines * 5)

    # Check for element mappings coverage
    cursor.execute("""
                   SELECT COUNT(DISTINCT em.v_query_code) as mapped_queries,
                          COUNT(DISTINCT pm.v_query_code) as total_queries
                   FROM etl_pipeline_metadata pm
                            LEFT JOIN etl_element_mapping em ON pm.v_query_code = em.v_query_code
                   WHERE pm.v_contract_code = ?
                   """, (contract_code,))
    mapping_coverage = cursor.fetchone()

    if mapping_coverage[1] > 0:
        coverage_ratio = mapping_coverage[0] / mapping_coverage[1] if mapping_coverage[1] > 0 else 0
        if coverage_ratio < 0.5:
            quality_issues.append(f"Low element mapping coverage: {coverage_ratio:.1%}")
            quality_score -= 20

    conn.close()

    # Ensure score doesn't go below 0
    quality_score = max(0, quality_score)

    # Determine quality level
    if quality_score >= 90:
        quality_level = "Excellent"
    elif quality_score >= 75:
        quality_level = "Good"
    elif quality_score >= 60:
        quality_level = "Fair"
    elif quality_score >= 40:
        quality_level = "Poor"
    else:
        quality_level = "Critical"

    return {
        "success": True,
        "contract_code": contract_code,
        "quality_score": quality_score,
        "quality_level": quality_level,
        "quality_issues": quality_issues,
        "pipeline_count": pipeline_count,
        "recommendations": [
            "Fill in missing metadata fields",
            "Provide detailed descriptions for all components",
            "Ensure element mapping coverage",
            "Regular quality audits and reviews"
        ]
    }


@tool
def get_contract_statistics(contract_code: Optional[str] = None) -> Dict[str, Any]:
    """Gets comprehensive statistics for a specific contract or all contracts."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    if contract_code:
        # Statistics for specific contract
        cursor.execute("""
                       SELECT dc.v_contract_code,
                              dc.v_contract_name,
                              COUNT(DISTINCT pm.v_query_code)          as pipeline_count,
                              COUNT(DISTINCT pm.v_source_type)         as unique_source_types,
                              COUNT(DISTINCT pm.v_target_type)         as unique_target_types,
                              COUNT(DISTINCT em.v_transformation_code) as transformation_count
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                                LEFT JOIN etl_element_mapping em ON pm.v_query_code = em.v_query_code
                       WHERE dc.v_contract_code = ?
                       GROUP BY dc.v_contract_code, dc.v_contract_name
                       """, (contract_code,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return {"success": False, "error": f"Contract {contract_code} not found"}

        stats = {
            "contract_code": result[0],
            "contract_name": result[1],
            "pipeline_count": result[2] or 0,
            "unique_source_types": result[3] or 0,
            "unique_target_types": result[4] or 0,
            "transformation_count": result[5] or 0
        }
    else:
        # Overall statistics
        cursor.execute("""
                       SELECT COUNT(DISTINCT dc.v_contract_code)       as total_contracts,
                              COUNT(DISTINCT pm.v_query_code)          as total_pipelines,
                              COUNT(DISTINCT pm.v_source_type)         as unique_source_types,
                              COUNT(DISTINCT pm.v_target_type)         as unique_target_types,
                              COUNT(DISTINCT em.v_transformation_code) as total_transformations
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                                LEFT JOIN etl_element_mapping em ON pm.v_query_code = em.v_query_code
                       """)
        result = cursor.fetchone()

        stats = {
            "total_contracts": result[0] or 0,
            "total_pipelines": result[1] or 0,
            "unique_source_types": result[2] or 0,
            "unique_target_types": result[3] or 0,
            "total_transformations": result[4] or 0
        }

    conn.close()
    return {"success": True, "statistics": stats}


@tool
def analyze_pipeline_distribution(contract_code: Optional[str] = None) -> Dict[str, Any]:
    """Analyzes pipeline distribution by source types, target types, and systems."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    where_clause = "WHERE pm.v_contract_code = ?" if contract_code else ""
    params = (contract_code,) if contract_code else ()

    # Source type distribution
    cursor.execute(f"""
        SELECT pm.v_source_type, COUNT(*) as count
        FROM etl_pipeline_metadata pm
        {where_clause}
        GROUP BY pm.v_source_type
        ORDER BY count DESC
    """, params)
    source_types = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]

    # Target type distribution
    cursor.execute(f"""
        SELECT pm.v_target_type, COUNT(*) as count
        FROM etl_pipeline_metadata pm
        {where_clause}
        GROUP BY pm.v_target_type
        ORDER BY count DESC
    """, params)
    target_types = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]

    # System analysis (if contract_code provided)
    system_analysis = {}
    if contract_code:
        cursor.execute("""
                       SELECT dc.v_source_system, dc.v_target_system, COUNT(pm.v_query_code) as pipeline_count
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                       WHERE dc.v_contract_code = ?
                       GROUP BY dc.v_source_system, dc.v_target_system
                       """, (contract_code,))
        result = cursor.fetchone()
        if result:
            system_analysis = {
                "source_system": result[0],
                "target_system": result[1],
                "pipeline_count": result[2] or 0
            }

    conn.close()

    return {
        "success": True,
        "source_type_distribution": source_types,
        "target_type_distribution": target_types,
        "system_analysis": system_analysis,
        "contract_code": contract_code
    }


@tool
def get_system_analysis(contract_code: Optional[str] = None) -> Dict[str, Any]:
    """Analyzes system landscape and data flow patterns."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    if contract_code:
        # Analysis for specific contract
        cursor.execute("""
                       SELECT dc.v_source_system,
                              dc.v_target_system,
                              COUNT(DISTINCT pm.v_query_code)             as pipeline_count,
                              COUNT(DISTINCT pm.v_source_table_or_object) as source_objects,
                              COUNT(DISTINCT pm.v_target_table_or_object) as target_objects
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                       WHERE dc.v_contract_code = ?
                       GROUP BY dc.v_source_system, dc.v_target_system
                       """, (contract_code,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return {"success": False, "error": f"Contract {contract_code} not found"}

        analysis = {
            "contract_code": contract_code,
            "source_system": result[0],
            "target_system": result[1],
            "pipeline_count": result[2] or 0,
            "source_objects": result[3] or 0,
            "target_objects": result[4] or 0,
            "data_flow_complexity": "High" if (result[2] or 0) > 3 else "Medium" if (result[2] or 0) > 1 else "Low"
        }
    else:
        # Overall system landscape analysis
        cursor.execute("""
                       SELECT dc.v_source_system,
                              COUNT(DISTINCT dc.v_contract_code) as contract_count,
                              COUNT(DISTINCT pm.v_query_code)    as pipeline_count
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                       GROUP BY dc.v_source_system
                       ORDER BY pipeline_count DESC
                       """)
        source_systems = [{"system": row[0], "contracts": row[1], "pipelines": row[2]} for row in cursor.fetchall()]

        cursor.execute("""
                       SELECT dc.v_target_system,
                              COUNT(DISTINCT dc.v_contract_code) as contract_count,
                              COUNT(DISTINCT pm.v_query_code)    as pipeline_count
                       FROM data_contracts dc
                                LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                       GROUP BY dc.v_target_system
                       ORDER BY pipeline_count DESC
                       """)
        target_systems = [{"system": row[0], "contracts": row[1], "pipelines": row[2]} for row in cursor.fetchall()]

        analysis = {
            "source_systems": source_systems,
            "target_systems": target_systems,
            "total_unique_source_systems": len(source_systems),
            "total_unique_target_systems": len(target_systems)
        }

    conn.close()
    return {"success": True, "system_analysis": analysis}


@tool
def get_all_contracts_summary() -> Dict[str, Any]:
    """Gets a comprehensive summary of all contracts with quality metrics."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT dc.v_contract_code,
                          dc.v_contract_name,
                          dc.v_contract_description,
                          dc.v_source_owner,
                          dc.v_ingestion_owner,
                          dc.v_source_system,
                          dc.v_target_system,
                          COUNT(DISTINCT pm.v_query_code)  as pipeline_count,
                          COUNT(DISTINCT pm.v_source_type) as source_types,
                          COUNT(DISTINCT pm.v_target_type) as target_types
                   FROM data_contracts dc
                            LEFT JOIN etl_pipeline_metadata pm ON dc.v_contract_code = pm.v_contract_code
                   GROUP BY dc.v_contract_code, dc.v_contract_name, dc.v_contract_description,
                            dc.v_source_owner, dc.v_ingestion_owner, dc.v_source_system, dc.v_target_system
                   ORDER BY pipeline_count DESC
                   """)

    contracts = []
    for row in cursor.fetchall():
        # Calculate basic quality score
        quality_score = 100
        missing_fields = 0

        fields = [row[2], row[3], row[4], row[5], row[6]]  # description, owners, systems
        for field in fields:
            if not field or field.strip() == "":
                missing_fields += 1
                quality_score -= 20

        contracts.append({
            "contract_code": row[0],
            "contract_name": row[1],
            "description": row[2] or "No description",
            "source_owner": row[3] or "Not specified",
            "ingestion_owner": row[4] or "Not specified",
            "source_system": row[5] or "Not specified",
            "target_system": row[6] or "Not specified",
            "pipeline_count": row[7] or 0,
            "source_types": row[8] or 0,
            "target_types": row[9] or 0,
            "quality_score": max(0, quality_score),
            "missing_fields": missing_fields
        })

    conn.close()

    # Calculate summary statistics
    total_contracts = len(contracts)
    avg_pipelines = sum(c["pipeline_count"] for c in contracts) / total_contracts if total_contracts > 0 else 0
    avg_quality = sum(c["quality_score"] for c in contracts) / total_contracts if total_contracts > 0 else 0

    contracts_with_issues = len([c for c in contracts if c["quality_score"] < 80])

    return {
        "success": True,
        "contracts": contracts,
        "summary": {
            "total_contracts": total_contracts,
            "average_pipelines_per_contract": round(avg_pipelines, 1),
            "average_quality_score": round(avg_quality, 1),
            "contracts_with_quality_issues": contracts_with_issues,
            "quality_issues_percentage": round((contracts_with_issues / total_contracts * 100),
                                               1) if total_contracts > 0 else 0
        }
    }


# Existing tools (keeping the same as before)
@tool
def query_pipeline_dependencies(query_codes: List[str]) -> Dict[str, Any]:
    """Gets downstream pipeline dependencies for a given list of query codes."""
    if not query_codes: return {"success": True, "dependencies": {}}
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in query_codes])
    cursor.execute(
        f"SELECT v_query_code, v_depends_on FROM etl_pipeline_dependency WHERE v_query_code IN ({placeholders})",
        query_codes)
    results = cursor.fetchall()
    conn.close()
    dependencies = {}
    for from_q, to_q in results:
        if from_q not in dependencies: dependencies[from_q] = []
        dependencies[from_q].append(to_q)
    return {"success": True, "dependencies": dependencies}


@tool
def query_element_mappings_by_queries(query_codes: List[str]) -> Dict[str, Any]:
    """Gets element mappings for specific query codes."""
    if not query_codes: return {"success": True, "mappings": []}
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in query_codes])
    query = f"""
        SELECT eem.v_query_code, eem.v_source_data_element_code, eem.v_target_data_element_code, tr.v_transformation_rules,
               src.v_data_element_name, src.v_table_name, tgt.v_data_element_name, tgt.v_table_name
        FROM etl_element_mapping eem
        JOIN business_element_mapping src ON eem.v_source_data_element_code = src.v_data_element_code
        JOIN business_element_mapping tgt ON eem.v_target_data_element_code = tgt.v_data_element_code
        JOIN transformation_rules tr ON eem.v_transformation_code = tr.v_transformation_code
        WHERE eem.v_query_code IN ({placeholders})
    """
    cursor.execute(query, query_codes)
    results = cursor.fetchall()
    conn.close()
    mappings = [{"query_code": r[0], "source_code": r[1], "target_code": r[2], "rules": r[3],
                 "source_name": r[4], "source_table": r[5], "target_name": r[6], "target_table": r[7]} for r in results]
    return {"success": True, "mappings": mappings}


@tool
def find_element_by_name(element_name: str) -> Dict[str, Any]:
    """Finds a data element by its name."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT v_data_element_code, v_data_element_name, v_table_name FROM business_element_mapping WHERE v_data_element_name LIKE ?",
        (f"%{element_name}%",))
    results = cursor.fetchall()
    conn.close()
    if results:
        elements = [{"element_code": r[0], "element_name": r[1], "table_name": r[2]} for r in results]
        return {"success": True, "elements": elements}
    return {"success": False, "error": f"Element '{element_name}' not found."}


@tool
def trace_element_connections(element_code: str, direction: str) -> Dict[str, Any]:
    """Traces connections for a data element."""
    connections = []
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    if direction in ['downstream', 'bidirectional']:
        cursor.execute(
            "SELECT v_target_data_element_code FROM etl_element_mapping WHERE v_source_data_element_code = ?",
            (element_code,))
        connections.extend([{"connected_code": r[0], "direction": "downstream"} for r in cursor.fetchall()])
    if direction in ['upstream', 'bidirectional']:
        cursor.execute(
            "SELECT v_source_data_element_code FROM etl_element_mapping WHERE v_target_data_element_code = ?",
            (element_code,))
        connections.extend([{"connected_code": r[0], "direction": "upstream"} for r in cursor.fetchall()])

    conn.close()
    return {"success": True, "connections": connections}


@tool
def get_all_query_codes() -> Dict[str, Any]:
    """Dynamically fetch all available query codes from database."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT v_query_code FROM etl_pipeline_metadata ORDER BY v_query_code")
    results = cursor.fetchall()
    conn.close()

    query_codes = [row[0] for row in results]
    return {"success": True, "query_codes": query_codes}


@tool
def get_available_contracts() -> Dict[str, Any]:
    """Dynamically fetch all available contracts from database."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts")
    results = cursor.fetchall()
    conn.close()

    contracts = [{"contract_code": r[0], "contract_name": r[1], "description": r[2]} for r in results]
    return {"success": True, "contracts": contracts}


@tool
def get_available_elements() -> Dict[str, Any]:
    """Dynamically fetch all available data elements from database."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT v_data_element_name, v_table_name FROM business_element_mapping ORDER BY v_data_element_name")
    results = cursor.fetchall()
    conn.close()

    elements = [{"element_name": r[0], "table_name": r[1]} for r in results]
    return {"success": True, "elements": elements}


def get_llm():
    """Helper function to get the LLM instance."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def contract_summarization_agent(state: LineageState):
    """New agent for comprehensive contract summarization and quality analysis."""
    logger.info("---EXECUTING: Contract Summarization Agent---")

    llm = get_llm()
    contract_name = state.get('contract_name', state.get('input_parameter'))

    try:
        # If no specific contract mentioned, provide overall summary
        if not contract_name or contract_name == state.get('input_parameter'):
            # Check if user wants overall statistics
            user_query_lower = state.get('input_parameter', '').lower()
            if any(keyword in user_query_lower for keyword in ['all contracts', 'overall', 'global', 'entire']):
                # Get overall summary
                all_contracts_summary = get_all_contracts_summary.invoke({})
                overall_stats = get_contract_statistics.invoke({})
                system_analysis = get_system_analysis.invoke({})

                if not all_contracts_summary.get("success"):
                    return {**state, "error_message": "Failed to retrieve contract summary", "current_step": "error"}

                # Generate comprehensive analysis
                analysis_context = f"""
                Overall Data Contract Ecosystem Analysis:

                Contract Summary: {json.dumps(all_contracts_summary.get('summary', {}), indent=2)}
                System Analysis: {json.dumps(system_analysis.get('system_analysis', {}), indent=2)}
                Overall Statistics: {json.dumps(overall_stats.get('statistics', {}), indent=2)}

                Contracts with Issues: {json.dumps([c for c in all_contracts_summary.get('contracts', []) if c['quality_score'] < 80], indent=2)}

                Please provide:
                1. Executive summary of the data contract ecosystem
                2. Key quality issues and recommendations
                3. System landscape analysis
                4. Risk assessment and governance recommendations
                """

                messages = [
                    SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
                    HumanMessage(content=analysis_context)
                ]

                response = llm.invoke(messages)

                # Compile final result
                final_result = {
                    "analysis_type": "overall_contract_ecosystem",
                    "summary": all_contracts_summary,
                    "statistics": overall_stats.get('statistics', {}),
                    "system_analysis": system_analysis.get('system_analysis', {}),
                    "llm_insights": response.content,
                    "timestamp": datetime.now().isoformat()
                }

                return {**state, "final_result": final_result, "current_step": "complete"}
            else:
                # Ask user to specify a contract
                available_contracts = get_available_contracts.invoke({})
                contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]

                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"Please specify which contract you'd like to analyze, or say 'all contracts' for overall analysis.\n\n"
                                              f"Available contracts:\n" +
                                              "\n".join([f"{i + 1}. {contract}" for i, contract in
                                                         enumerate(contract_names)]) +
                                              f"\n\nOr type 'all contracts' for ecosystem-wide analysis.",
                    "query_results": {"available_contracts": [{"index": i, "name": contract} for i, contract in
                                                              enumerate(contract_names)]},
                }

        # Get contract details
        contract_details = query_contract_by_name.invoke({"contract_name": contract_name})
        if not contract_details.get("success"):
            return {**state, "error_message": contract_details.get("error"), "current_step": "error"}

        contract_code = contract_details["contract_code"]

        # Gather comprehensive contract data
        quality_analysis = analyze_contract_quality.invoke({"contract_code": contract_code})
        contract_statistics = get_contract_statistics.invoke({"contract_code": contract_code})
        pipeline_distribution = analyze_pipeline_distribution.invoke({"contract_code": contract_code})
        system_analysis = get_system_analysis.invoke({"contract_code": contract_code})
        pipelines = query_pipelines_by_contract.invoke({"contract_code": contract_code})

        # Get dependencies for better analysis
        query_codes = [p['query_code'] for p in pipelines.get("pipelines", [])]
        dependencies = query_pipeline_dependencies.invoke({"query_codes": query_codes})
        mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})

        # Use LLM to generate comprehensive insights
        analysis_context = f"""
        Comprehensive Contract Analysis for: {contract_name}

        Contract Details: {json.dumps(contract_details, indent=2)}
        Quality Analysis: {json.dumps(quality_analysis, indent=2)}
        Statistics: {json.dumps(contract_statistics, indent=2)}
        Pipeline Distribution: {json.dumps(pipeline_distribution, indent=2)}
        System Analysis: {json.dumps(system_analysis, indent=2)}
        Pipeline Dependencies: {json.dumps(dependencies, indent=2)}
        Element Mappings Count: {len(mappings.get('mappings', []))}

        Please provide a comprehensive analysis including:
        1. Contract quality assessment and specific issues found
        2. Pipeline architecture and complexity analysis
        3. Data flow patterns and system integration analysis
        4. Risk assessment and data governance recommendations
        5. Specific action items for improvement
        6. Comparison with industry best practices
        """

        messages = [
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(content=analysis_context)
        ]

        response = llm.invoke(messages)

        # Compile comprehensive results
        final_result = {
            "analysis_type": "contract_summarization",
            "contract_name": contract_name,
            "contract_code": contract_code,
            "contract_details": contract_details,
            "quality_analysis": quality_analysis,
            "statistics": contract_statistics.get('statistics', {}),
            "pipeline_distribution": pipeline_distribution,
            "system_analysis": system_analysis.get('system_analysis', {}),
            "pipeline_dependencies": dependencies,
            "element_mapping_count": len(mappings.get('mappings', [])),
            "pipelines": pipelines.get("pipelines", []),
            "llm_insights": response.content,
            "recommendations": [
                "Address identified quality issues",
                "Improve metadata documentation",
                "Enhance element mapping coverage",
                "Regular quality audits"
            ],
            "timestamp": datetime.now().isoformat()
        }

        return {**state, "final_result": final_result, "current_step": "complete"}

    except Exception as e:
        logger.error(f"Error in contract summarization: {e}")
        traceback.print_exc()
        return {**state, "error_message": f"Contract summarization failed: {str(e)}", "current_step": "error"}


# Keep all existing agents (contract_analysis_agent, element_analysis_agent, etc.)
def contract_analysis_agent(state: LineageState):
    """LLM-powered contract analysis with intelligent insights."""
    logger.info("---EXECUTING: Intelligent Contract Analysis---")

    llm = get_llm()
    contract_name = state.get('contract_name', state.get('input_parameter'))

    # Gather contract data
    contract_details = query_contract_by_name.invoke({"contract_name": contract_name})
    if not contract_details.get("success"):
        return {**state, "error_message": contract_details.get("error"), "current_step": "error"}

    contract_code = contract_details["contract_code"]
    pipelines = query_pipelines_by_contract.invoke({"contract_code": contract_code})
    query_codes = [p['query_code'] for p in pipelines.get("pipelines", [])]
    dependencies = query_pipeline_dependencies.invoke({"query_codes": query_codes})
    mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})

    # Use LLM to analyze the contract data
    analysis_context = f"""
    Contract Analysis:
    - Contract: {contract_details}
    - Pipelines: {pipelines.get("pipelines", [])}
    - Dependencies: {dependencies.get("dependencies", {})}
    - Element Mappings: {mappings.get("mappings", [])}

    Please analyze this contract and provide:
    1. Key insights about the data flow
    2. Potential bottlenecks or issues
    3. Data quality considerations
    4. Recommendations for optimization
    5. Complexity assessment (1-10)
    """

    messages = [
        SystemMessage(content=CONTRACT_SYSTEM_PROMPT),
        HumanMessage(content=analysis_context)
    ]

    try:
        response = llm.invoke(messages)

        # Extract complexity score and recommendations from LLM response
        complexity_score = 5  # Default, could be parsed from LLM response
        recommendations = [
            "Consider monitoring data quality in transformation steps",
            "Review pipeline dependencies for optimization opportunities"
        ]

        state["query_results"].update({
            "contract": contract_details,
            "pipelines": pipelines.get("pipelines", []),
            "dependencies": dependencies.get("dependencies", {}),
            "mappings": mappings.get("mappings", [])
        })

        state["llm_analysis"] = {
            "contract_insights": response.content,
            "complexity_assessment": complexity_score
        }
        state["recommendations"] = recommendations
        state["complexity_score"] = complexity_score

        return {**state, "current_step": "lineage_construction"}

    except Exception as e:
        logger.error(f"Error incoordinatoragent: {e}")
    traceback.print_exc()
    return {**state, "error_message": f"Coordinator analysis failed: {str(e)}", "current_step": "error"}

def lineage_coordinator_agent(state: LineageState):
    """Enhanced coordinator that handles contract summarization requests."""
    logger.info("---EXECUTING: Enhanced Lineage Coordinator---")

    llm = get_llm()

    # Get the latest human message
    user_query = state.get('input_parameter', '')

    # Dynamically fetch available data from database
    available_contracts = get_available_contracts.invoke({})
    available_elements = get_available_elements.invoke({})

    contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]
    element_names = [e['element_name'] for e in available_elements.get('elements', [])]

    context = f"""
    User Query: {user_query}

    Current State:
    - Messages: {len(state.get('messages', []))} messages in conversation
    - Previous context: {state.get('query_results', {})}

    Available Information in Database:
    - Data Contracts: {contract_names}
    - Data Elements: {element_names}

    Analysis Rules:
    - If query mentions "summarize", "summary", "analyze contract", "quality issues", "contract analysis", "statistics" -> contract_summarization
    - If query mentions specific element names from available elements -> element_based
    - If query mentions "contract", "pipeline" or contract names -> contract_based
    - If query asks to "trace [element_name]" -> element_based
    - Default traversal is bidirectional unless specified
    """

    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{context}\n\nPlease analyze this request and determine:"
                    f"\n1. Lineage type (contract_based, element_based, or contract_summarization)\n"
                    f"2. Key parameters to extract\n"
                    f"3. Traversal direction\n"
                    f"4. Any clarifications needed\n\nFor element queries, identify the specific element name to search for.")
    ]

    try:
        response = llm.invoke(messages)

        # Parse LLM response for decision making
        analysis = {
            "llm_reasoning": response.content,
            "confidence_score": 0.8
        }

        content = response.content.lower()
        user_query_lower = user_query.lower()

        # Check for contract summarization keywords
        summarization_keywords = ['summarize', 'summary', 'analyze contract', 'quality issues', 'contract analysis',
                                  'statistics', 'quality', 'metadata issues', 'pipeline count', 'system analysis']

        if any(keyword in user_query_lower for keyword in summarization_keywords):
            lineage_type = LineageType.CONTRACT_SUMMARIZATION.value

            # Check if specific contract mentioned
            mentioned_contract = None
            for contract in contract_names:
                contract_words = contract.lower().split()
                if any(word in user_query_lower for word in contract_words):
                    mentioned_contract = contract
                    break

            if mentioned_contract:
                state['contract_name'] = mentioned_contract

            next_step = "contract_summarization"

        else:
            # Dynamic element detection from database
            mentioned_element = None
            for element in element_names:
                if element.lower() in user_query_lower:
                    mentioned_element = element
                    break

            # Dynamic contract detection from database
            mentioned_contract = None
            for contract in contract_names:
                contract_words = contract.lower().split()
                if any(word in user_query_lower for word in contract_words):
                    mentioned_contract = contract
                    break

            # Determine lineage type based on content analysis
            if mentioned_element or (
                    'trace' in user_query_lower and any(elem.lower() in user_query_lower for elem in element_names)):
                lineage_type = LineageType.ELEMENT_BASED.value

                # Extract element name from query
                element_name = mentioned_element
                if not element_name:
                    # Try to extract from "trace X lineage" pattern
                    import re
                    match = re.search(r'trace\s+(\w+)', user_query_lower)
                    if match:
                        potential_element = match.group(1)
                        # Check if it matches any available element
                        for elem in element_names:
                            if elem.lower().startswith(potential_element) or potential_element in elem.lower():
                                element_name = elem
                                break

                # If no element found or identified, trigger HITL
                if not element_name:
                    return {
                        **state,
                        "requires_human_approval": True,
                        "human_approval_message": f"I detected this might be an element-based query, but couldn't identify the specific element from your request '{user_query}'.\n\n"
                                                  f"Please select from available elements:\n" +
                                                  "\n".join(
                                                      [f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
                                                  f"\n\nOr rephrase your query to be more specific.",
                        "query_results": {
                            "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]},
                        "llm_analysis": analysis
                    }

                # Search for the element in database
                element_search_result = find_element_by_name.invoke({"element_name": element_name})

                if not element_search_result.get("success"):
                    # Trigger HITL for element selection
                    return {
                        **state,
                        "requires_human_approval": True,
                        "human_approval_message": f"Could not find element matching '{element_name}' in the database.\n\n"
                                                  f"Available elements are:\n" +
                                                  "\n".join(
                                                      [f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
                                                  f"\n\nPlease select the correct element number or provide a more specific name.",
                        "query_results": {
                            "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]},
                        "llm_analysis": analysis,
                        "error_message": f"Element '{element_name}' not found"
                    }

                found_elements = element_search_result["elements"]
                if len(found_elements) > 1:
                    # Enhanced HITL with LLM-generated clarification
                    clarification_prompt = f"""
                    Based on the user query "{user_query}", I found multiple matching elements:
                    {json.dumps(found_elements, indent=2)}

                    Generate a clear, helpful message asking the user to select the correct element.
                    Include context about why each option might be relevant.
                    """

                    clarify_messages = [
                        SystemMessage(content=APPROVAL_SYSTEM_PROMPT),
                        HumanMessage(content=clarification_prompt)
                    ]

                    clarify_response = llm.invoke(clarify_messages)

                    return {
                        **state,
                        "requires_human_approval": True,
                        "human_approval_message": clarify_response.content,
                        "query_results": {"ambiguous_elements": found_elements},
                        "llm_analysis": analysis
                    }

                # Set up for element tracing
                state['element_queue'] = [found_elements[0]['element_code']]
                state['traced_elements'] = set()
                state['element_name'] = element_name
                next_step = "element_analysis"

            elif mentioned_contract or any(
                    keyword in content or keyword in user_query_lower for keyword in ['contract', 'pipeline']):
                lineage_type = LineageType.CONTRACT_BASED.value

                # Extract contract name from query
                contract_name = mentioned_contract if mentioned_contract else user_query

                # If no specific contract mentioned, trigger HITL
                if not mentioned_contract:
                    return {
                        **state,
                        "requires_human_approval": True,
                        "human_approval_message": f"I detected this is a contract-based query, but couldn't identify the specific contract from your request '{user_query}'.\n\n"
                                                  f"Available contracts are:\n" +
                                                  "\n".join([f"{i + 1}. {contract}" for i, contract in
                                                             enumerate(contract_names)]) +
                                                  f"\n\nPlease select the contract number or provide a more specific name.",
                        "query_results": {"available_contracts": [{"index": i, "name": contract} for i, contract in
                                                                  enumerate(contract_names)]},
                        "llm_analysis": analysis
                    }

                state['contract_name'] = contract_name
                next_step = "contract_analysis"
            else:
                # LLM couldn't determine - ask for clarification with dynamic options
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"""I need clarification about your request '{user_query}'. 
                    Are you looking for:

                    1. **Contract Summarization** - Analyze contract quality, statistics, and metadata issues
                       Example: "Summarize Customer Data Pipeline" or "Analyze contract quality"

                    2. **Element-based lineage** - Trace a specific data field
                       Available elements: {', '.join(element_names[:10])}{'...' if len(element_names) > 10 else ''}

                    3. **Contract-based lineage** - Analyze a data pipeline
                       Available contracts: {', '.join(contract_names)}

                    Please specify which type of analysis you need and be more specific about what you want to trace.""",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)],
                        "available_contracts": [{"index": i, "name": contract} for i, contract in
                                                enumerate(contract_names)]
                    },
                    "llm_analysis": analysis
                }

        # Determine traversal direction using LLM
        if 'upstream' in content or 'upstream' in user_query_lower:
            direction = TraversalDirection.UPSTREAM.value
        elif 'downstream' in content or 'downstream' in user_query_lower:
            direction = TraversalDirection.DOWNSTREAM.value
        else:
            direction = TraversalDirection.BIDIRECTIONAL.value

        logger.info(f"Coordinator decision: type={lineage_type}, direction={direction}, next_step={next_step}")

        return {
            **state,
            "lineage_type": lineage_type,
            "traversal_direction": direction,
            "current_step": next_step,
            "llm_analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error in contract analysis: {e}")
        return {**state, "error_message": f"Contract analysis failed: {str(e)}", "current_step": "error"}


def element_analysis_agent(state: LineageState):
    """Intelligent element tracing with LLM-powered insights."""
    logger.info("---EXECUTING: Intelligent Element Analysis---")

    llm = get_llm()

    if not state.get("element_queue"):
        return {**state, "current_step": "lineage_construction"}

    element_to_trace = state["element_queue"].pop(0)

    if element_to_trace in state["traced_elements"]:
        return {**state}

    state["traced_elements"].add(element_to_trace)

    # Dynamically get all available query codes from database
    all_query_codes_result = get_all_query_codes.invoke({})
    if not all_query_codes_result.get("success"):
        logger.warning("Failed to fetch query codes, using fallback")
        query_codes = ["Q001", "Q002", "Q003"]  # Fallback
    else:
        query_codes = all_query_codes_result["query_codes"]

    # Get element mappings with LLM analysis using dynamic query codes
    all_mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})
    related_mappings = [m for m in all_mappings.get("mappings", [])
                        if m['source_code'] == element_to_trace or m['target_code'] == element_to_trace]

    # Use LLM to analyze transformations and provide insights
    element_context = f"""
    Element Analysis for: {element_to_trace}
    Related Mappings: {json.dumps(related_mappings, indent=2)}

    Please analyze:
    1. Data transformation patterns
    2. Business logic implications
    3. Data quality risks
    4. Governance considerations
    5. Suggest next elements to trace
    """

    messages = [
        SystemMessage(content=ELEMENT_SYSTEM_PROMPT),
        HumanMessage(content=element_context)
    ]

    try:
        response = llm.invoke(messages)

        # Initialize mappings list if it doesn't exist and add only new mappings
        if "mappings" not in state["query_results"]:
            state["query_results"]["mappings"] = []

        # Get existing mappings to avoid duplication
        existing_mappings = state["query_results"]["mappings"]
        existing_mapping_keys = set()
        for em in existing_mappings:
            key = (em['query_code'], em['source_code'], em['target_code'])
            existing_mapping_keys.add(key)

        # Add only new mappings
        new_mappings = []
        for m in related_mappings:
            key = (m['query_code'], m['source_code'], m['target_code'])
            if key not in existing_mapping_keys:
                new_mappings.append(m)
                existing_mapping_keys.add(key)

        state["query_results"]["mappings"].extend(new_mappings)

        # Add LLM insights
        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"][f"element_{element_to_trace}"] = response.content

        # Add newly discovered elements to queue
        for m in related_mappings:
            if m['source_code'] == element_to_trace and m['target_code'] not in state["traced_elements"]:
                state["element_queue"].append(m['target_code'])
            if m['target_code'] == element_to_trace and m['source_code'] not in state["traced_elements"]:
                state["element_queue"].append(m['source_code'])

        return {**state}

    except Exception as e:
        logger.error(f"Error in element analysis: {e}")
        return {**state, "error_message": f"Element analysis failed: {str(e)}", "current_step": "error"}


def lineage_construction_agent(state: LineageState):
    """LLM-powered lineage graph construction with optimization."""
    logger.info("---EXECUTING: Intelligent Lineage Construction---")

    llm = get_llm()

    # Build graph structure
    nodes, edges = [], []
    unique_nodes = {}
    unique_edges = set()

    mappings = state.get("query_results", {}).get("mappings", [])
    for m in mappings:
        # Add source node
        src_key = f"{m['source_table']}.{m['source_name']}"
        if src_key not in unique_nodes:
            unique_nodes[src_key] = {
                "id": m['source_code'],
                "name": m['source_name'],
                "table": m['source_table'],
                "type": "source",
                "element_code": m['source_code']
            }

        # Add target node
        tgt_key = f"{m['target_table']}.{m['target_name']}"
        if tgt_key not in unique_nodes:
            unique_nodes[tgt_key] = {
                "id": m['target_code'],
                "name": m['target_name'],
                "table": m['target_table'],
                "type": "target",
                "element_code": m['target_code']
            }

        # adding only unique edges
        edge_key = (m['source_code'], m['target_code'], m['query_code'])
        if edge_key not in unique_edges:
            edges.append({
                "source": m['source_code'],
                "target": m['target_code'],
                "transformation": m['rules'],
                "query_code": m['query_code']
            })
            unique_edges.add(edge_key)

    nodes = list(unique_nodes.values())

    # Use LLM to analyze graph structure and provide insights
    graph_context = f"""
    Lineage Graph Analysis:
    - Total Nodes: {len(nodes)}
    - Total Edges: {len(edges)}
    - Node Details: {json.dumps(nodes, indent=2)[:1000]}...
    - Edge Details: {json.dumps(edges, indent=2)[:1000]}...

    Please analyze this lineage graph and provide:
    1. Graph complexity assessment (1-10)
    2. Critical path identification
    3. Transformation complexity analysis
    4. Recommendations for graph optimization
    5. Potential data governance issues
    """

    messages = [
        SystemMessage(content=CONSTRUCTION_SYSTEM_PROMPT),
        HumanMessage(content=graph_context)
    ]

    try:
        response = llm.invoke(messages)

        # Update state with constructed graph and LLM analysis
        state["lineage_nodes"] = nodes
        state["lineage_edges"] = edges

        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["graph_construction"] = response.content

        # Extract complexity score from analysis (simplified)
        complexity_score = len(edges) + len(nodes)  # Basic calculation, could be enhanced
        state["complexity_score"] = min(complexity_score // 2, 10)

        return {**state, "current_step": "finalize_results"}

    except Exception as e:
        logger.error(f"Error in lineage construction: {e}")
        return {**state, "error_message": f"Graph construction failed: {str(e)}", "current_step": "error"}


def deduplicate_mappings(mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate mappings based on query_code, source_code, and target_code combination.
    """
    seen = set()
    unique_mappings = []

    for mapping in mappings:
        key = (mapping['query_code'], mapping['source_code'], mapping['target_code'])
        if key not in seen:
            unique_mappings.append(mapping)
            seen.add(key)

    return unique_mappings


def deduplicate_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate edges based on source, target, and query_code combination.
    """
    seen = set()
    unique_edges = []

    for edge in edges:
        key = (edge['source'], edge['target'], edge['query_code'])
        if key not in seen:
            unique_edges.append(edge)
            seen.add(key)

    return unique_edges


def human_approval_agent(state: LineageState):
    """Enhanced human interaction with LLM-powered response processing."""
    logger.info("---EXECUTING: Human Approval Agent---")

    # If no human approval required, continue
    if not state.get("requires_human_approval"):
        return {**state, "current_step": "error"}

    # Check if we have human feedback to process
    human_feedback = state.get("human_feedback")
    if not human_feedback:
        # No feedback yet - return state as is (workflow should be paused)
        logger.info("Waiting for human feedback...")
        return state

    logger.info(f"Processing human feedback: {human_feedback}")

    try:
        # Process based on the type of approval needed
        query_results = state.get("query_results", {})

        # Handle element disambiguation (multiple matching elements found)
        if "ambiguous_elements" in query_results:
            selected_index = human_feedback.get("selected_index")
            if selected_index is not None:
                elements = query_results["ambiguous_elements"]
                if 0 <= selected_index < len(elements):
                    selected_element = elements[selected_index]

                    # Set up for element analysis
                    return {
                        **state,
                        "element_queue": [selected_element["element_code"]],
                        "traced_elements": set(),
                        "element_name": selected_element["element_name"],
                        "lineage_type": LineageType.ELEMENT_BASED.value,
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "element_analysis"
                    }

        # Handle element selection from available options
        elif "available_elements" in query_results:
            selected_index = human_feedback.get("selected_index")
            selected_name = human_feedback.get("selected_name")

            available_elements = query_results["available_elements"]
            selected_element_name = None

            if selected_index is not None and 0 <= selected_index < len(available_elements):
                selected_element_name = available_elements[selected_index]["name"]
            elif selected_name:
                # Find by name
                for elem in available_elements:
                    if elem["name"].lower() == selected_name.lower():
                        selected_element_name = elem["name"]
                        break

            if selected_element_name:
                # Search for the element in database
                element_search_result = find_element_by_name.invoke({"element_name": selected_element_name})

                if element_search_result.get("success") and element_search_result["elements"]:
                    found_elements = element_search_result["elements"]

                    return {
                        **state,
                        "element_queue": [found_elements[0]["element_code"]],
                        "traced_elements": set(),
                        "element_name": selected_element_name,
                        "lineage_type": LineageType.ELEMENT_BASED.value,
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "element_analysis"
                    }

        # Handle contract selection from available options
        elif "available_contracts" in query_results:
            selected_index = human_feedback.get("selected_index")
            selected_name = human_feedback.get("selected_name")

            available_contracts = query_results["available_contracts"]
            selected_contract_name = None

            if selected_index is not None and 0 <= selected_index < len(available_contracts):
                selected_contract_name = available_contracts[selected_index]["name"]
            elif selected_name:
                # Check for special keywords first
                if selected_name.lower() in ['all contracts', 'overall', 'global', 'entire']:
                    return {
                        **state,
                        "contract_name": None,  # Signal for overall analysis
                        "lineage_type": LineageType.CONTRACT_SUMMARIZATION.value,
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "contract_summarization"
                    }

                # Find by name
                for contract in available_contracts:
                    if contract["name"].lower() == selected_name.lower():
                        selected_contract_name = contract["name"]
                        break

            if selected_contract_name:
                # Determine the correct step based on lineage type
                current_lineage_type = state.get("lineage_type", "")
                if current_lineage_type == LineageType.CONTRACT_SUMMARIZATION.value:
                    next_step = "contract_summarization"
                else:
                    next_step = "contract_analysis"

                return {
                    **state,
                    "contract_name": selected_contract_name,
                    "requires_human_approval": False,
                    "human_feedback": None,
                    "current_step": next_step
                }

        # If we reach here, the selection was invalid
        logger.warning("Invalid human feedback received")
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": (
                "Invalid selection. Please provide either:\n"
                "• A valid number from the options (e.g., 1, 2, 3)\n"
                "• The exact name of the item you want to select\n"
                "• For contract analysis, you can also say 'all contracts' for overall analysis\n\n"
                f"Original message: {state.get('human_approval_message', '')}"
            ),
            "human_feedback": None  # Clear invalid feedback
        }

    except Exception as e:
        logger.error(f"Error processing human feedback: {e}")
        traceback.print_exc()
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": f"Error processing your input: {str(e)}\n\nPlease try again.",
            "human_feedback": None
        }


def finalize_results_agent(state: LineageState):
    """Final result compilation with LLM-generated summary."""
    logger.info("---EXECUTING: Result Finalization---")

    llm = get_llm()

    # Remove duplicates from final results
    final_mappings = deduplicate_mappings(state.get("query_results", {}).get("mappings", []))
    final_edges = deduplicate_edges(state.get("lineage_edges", []))

    # Update state with cleaned data
    if "query_results" not in state:
        state["query_results"] = {}
    state["query_results"]["mappings"] = final_mappings
    state["lineage_edges"] = final_edges

    # Compile comprehensive results
    final_result = {
        "lineage_type": state.get("lineage_type"),
        "input_parameter": state.get("input_parameter"),
        "traversal_direction": state.get("traversal_direction"),
        "nodes": state.get("lineage_nodes", []),
        "edges": final_edges,
        "complexity_score": state.get("complexity_score", 0),
        "recommendations": state.get("recommendations", []),
        "query_results": {
            **state.get("query_results", {}),
            "mappings": final_mappings
        },
        "timestamp": datetime.now().isoformat()
    }

    # Generate executive summary with LLM
    summary_context = f"""
    Lineage Analysis Complete:
    - Analysis Type: {state.get("lineage_type")}
    - Input: {state.get("input_parameter")}
    - Nodes Found: {len(state.get("lineage_nodes", []))}
    - Relationships: {len(final_edges)}
    - Complexity Score: {state.get("complexity_score", 0)}/10
    - LLM Insights: {state.get("llm_analysis", {})}

    Generate a comprehensive executive summary including:
    1. Key findings and insights
    2. Data flow patterns discovered
    3. Risk assessment and recommendations
    4. Next steps for data governance
    """

    messages = [
        SystemMessage(
            content="You are an expert data analyst creating executive summaries of lineage analysis results."),
        HumanMessage(content=summary_context)
    ]

    try:
        response = llm.invoke(messages)
        final_result["executive_summary"] = response.content

        return {**state, "final_result": final_result, "current_step": "complete"}

    except Exception as e:
        logger.error(f"Error in result finalization: {e}")
        final_result["executive_summary"] = "Summary generation failed, but analysis completed successfully."
        return {**state, "final_result": final_result, "current_step": "complete"}


def error_handler_agent(state: LineageState):
    """Enhanced error handling with recovery suggestions."""
    logger.info("---EXECUTING: Error Handler---")

    error_msg = state.get("error_message", "Unknown error occurred")

    # Create error response with recovery suggestions
    error_result = {
        "success": False,
        "error": error_msg,
        "suggestions": [
            "Check if the input parameter exists in the database",
            "Try using more specific search terms",
            "Verify database connectivity and schema"
        ],
        "timestamp": datetime.now().isoformat()
    }

    return {**state, "final_result": error_result, "current_step": "complete"}


# --- Route Decision Functions ---
def should_continue(state: LineageState):
    """Enhanced routing to handle contract summarization."""
    current_step = state.get("current_step", "")

    if current_step == "error":
        return "error_handler"
    elif state.get("requires_human_approval"):
        return "human_approval"
    elif current_step == "contract_summarization":
        return "contract_summarization"
    elif current_step == "contract_analysis":
        return "contract_analysis"
    elif current_step == "element_analysis":
        return "element_analysis"
    elif current_step == "lineage_construction":
        return "lineage_construction"
    elif current_step == "finalize_results":
        return "finalize_results"
    elif current_step == "complete":
        return END
    else:
        return "coordinator"


def element_continue_condition(state: LineageState):
    """Decides whether to continue element tracing."""
    if state.get("element_queue") and len(state.get("element_queue", [])) > 0:
        return "element_analysis"
    else:
        return "lineage_construction"


# Enhanced Lineage Orchestrator
class LineageOrchestrator:
    """Enhanced orchestrator class for the intelligent lineage analysis workflow with contract summarization."""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.graph = self._build_workflow()
        # Store the complete workflow state instead of just config
        self.paused_state = None
        self.workflow_config = None

    def _build_workflow(self):
        """Build the enhanced LangGraph workflow with contract summarization."""
        workflow = StateGraph(LineageState)

        # Add all agent nodes including the new summarization agent
        workflow.add_node("coordinator", lineage_coordinator_agent)
        workflow.add_node("contract_summarization", contract_summarization_agent)
        workflow.add_node("contract_analysis", contract_analysis_agent)
        workflow.add_node("element_analysis", element_analysis_agent)
        workflow.add_node("lineage_construction", lineage_construction_agent)
        workflow.add_node("human_approval", human_approval_agent)
        workflow.add_node("finalize_results", finalize_results_agent)
        workflow.add_node("error_handler", error_handler_agent)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional routing
        workflow.add_conditional_edges("coordinator", should_continue)
        workflow.add_conditional_edges("contract_summarization", should_continue)
        workflow.add_conditional_edges("contract_analysis", should_continue)
        workflow.add_conditional_edges("element_analysis", element_continue_condition)
        workflow.add_conditional_edges("lineage_construction", should_continue)
        workflow.add_conditional_edges("human_approval", should_continue)
        workflow.add_conditional_edges("finalize_results", should_continue)
        workflow.add_edge("error_handler", END)

        # Compile with interrupt before human_approval
        return workflow.compile(interrupt_before=["human_approval"])

    async def execute_lineage_request(self, request: LineageRequest) -> Dict[str, Any]:
        """
        Execute a lineage analysis request with enhanced contract summarization capabilities.
        Returns either the final result or indicates human input is needed.
        """
        # Reset paused state
        self.paused_state = None
        self.workflow_config = None

        try:
            initial_state = {
                "messages": [HumanMessage(content=request.query)],
                "input_parameter": request.query,
                "lineage_type": "",
                "contract_name": None,
                "element_name": None,
                "traversal_direction": "bidirectional",
                "current_step": "start",
                "query_results": {},
                "lineage_nodes": [],
                "lineage_edges": [],
                "element_queue": [],
                "traced_elements": set(),
                "requires_human_approval": False,
                "human_approval_message": None,
                "human_feedback": None,
                "llm_analysis": {},
                "recommendations": [],
                "complexity_score": None,
                "contract_summary": None,
                "quality_assessment": None,
                "statistical_analysis": None,
                "final_result": None,
                "error_message": None
            }

            # Create a unique thread ID for this workflow execution
            thread_id = f"thread_{id(request)}"
            config = {"configurable": {"thread_id": thread_id}}

            self.workflow_config = config

            final_state = None
            async for event in self.graph.astream(initial_state, config):
                logger.info(f"Processing event: {list(event.keys())}")

                # Get the state from the event
                for node_name, state_data in event.items():
                    if isinstance(state_data, tuple):
                        if len(state_data) > 1 and isinstance(state_data[1], dict):
                            state_data = state_data[1]
                        else:
                            logger.warning(f"Unexpected tuple format in state_data: {state_data}")
                            continue

                    # Ensure state_data is a dictionary before proceeding
                    if not isinstance(state_data, dict):
                        logger.warning(f"state_data is not a dict: {type(state_data)}")
                        continue

                    final_state = state_data

                    # Check if human approval is required
                    if state_data.get("requires_human_approval"):
                        logger.info("---WORKFLOW PAUSED FOR HUMAN INPUT---")
                        # Store the current state for resumption
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "message": state_data.get("human_approval_message", "Human input required."),
                            "query_results": state_data.get("query_results", {}),
                            "thread_id": thread_id
                        }

            # If we reach here, workflow completed without human intervention
            if final_state and final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return {"error": "Workflow completed but no final result generated"}

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"Workflow execution failed: {str(e)}"}

    async def resume_with_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Resumes a paused workflow by providing human feedback."""
        if not self.paused_state or not self.workflow_config:
            return {"error": "No paused workflow to resume. Please start a new request."}

        try:
            logger.info(f"Resuming workflow with feedback: {feedback}")

            # Update the paused state with human feedback
            updated_state = {**self.paused_state}
            updated_state["human_feedback"] = feedback
            # true for human_approval agent processes it
            updated_state["requires_human_approval"] = True

            # Resume from the human_approval node
            final_state = None
            async for event in self.graph.astream(updated_state, self.workflow_config):
                logger.info(f"Resume event: {list(event.keys())}")

                for node_name, state_data in event.items():
                    if isinstance(state_data, tuple):
                        # in case of tuple, actual state is second element
                        if len(state_data) > 1 and isinstance(state_data[1], dict):
                            state_data = state_data[1]
                        else:
                            logger.warning(f"Unexpected tuple format in state_data: {state_data}")
                            continue

                    # Ensure state_data is a dictionary before proceeding
                    if not isinstance(state_data, dict):
                        logger.warning(f"state_data is not a dict: {type(state_data)}")
                        continue

                    final_state = state_data

                    # Check if another human approval is needed (in case of invalid input)
                    if (state_data.get("requires_human_approval") and
                            state_data.get("human_feedback") is None):
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "message": state_data.get("human_approval_message", "Additional input required."),
                            "query_results": state_data.get("query_results", {})
                        }

            # Clear paused state after successful completion
            self.paused_state = None
            self.workflow_config = None

            if final_state and final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return {"error": "Workflow resumed but no final result generated"}

        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            traceback.print_exc()
            return {"error": f"Failed to resume workflow: {str(e)}"}


async def main():
    """Enhanced main function with contract summarization examples."""
    orchestrator = LineageOrchestrator()

    print("Enhanced Data Lineage Agent with Contract Summarization")
    print("=" * 70)
    print("Example queries:")
    print("• 'trace customer_id' - Element-based lineage")
    print("• 'show lineage for Customer Data Pipeline' - Contract-based lineage")
    print("• 'summarize Customer Data Pipeline' - Contract quality analysis")
    print("• 'analyze contract quality issues' - Overall contract analysis")
    print("• 'show all contract statistics' - Ecosystem-wide analysis")

    while True:
        query = input("\nEnter your lineage query (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            continue

        request = LineageRequest(query=query)
        print("\nProcessing your request...")

        # Initial call to the orchestrator
        result = await orchestrator.execute_lineage_request(request)

        # Loop to handle human feedback until the workflow is complete
        while result.get("human_input_required"):
            print("\n" + "=" * 25)
            print("HUMAN INTERVENTION REQUIRED")
            print(result.get("message"))
            print("=" * 25)

            # Get user feedback
            user_response_str = input("Your response (e.g., a number or name): ").strip()

            # Try to convert to a number for index-based selection
            try:
                selected_index = int(user_response_str) - 1
                feedback = {"selected_index": selected_index}
            except ValueError:
                feedback = {"selected_name": user_response_str}

            print("\nResuming workflow with your feedback...")
            result = await orchestrator.resume_with_feedback(feedback)

        # Once the loop exits, we have the final result
        print("\n" + "=" * 70)
        print("FINAL ANALYSIS RESULTS")
        print("=" * 70)
        print(json.dumps(result, indent=2, default=str))


# --- Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())



