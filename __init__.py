import json
import asyncio
import random
import traceback
import os
from typing import Dict, List, Any, Optional, Set, Tuple, Annotated
from dataclasses import dataclass
from enum import Enum
import sqlite3
from datetime import datetime
import logging

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

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
2. Determine the most appropriate lineage strategy (contract-based or element-based)
3. Extract key parameters from natural language requests
4. Handle ambiguous requests by asking clarifying questions
5. Route the request to the appropriate specialized agent

Guidelines:
- If user mentions a specific contract name, pipeline, or data flow, choose contract-based lineage
- If user mentions specific data elements, fields, or columns, choose element-based lineage  
- If the request is ambiguous, ask for clarification
- Always extract the traversal direction (upstream, downstream, or bidirectional) from context
- Be conversational and helpful in your responses

Current request analysis: Determine the lineage type, extract parameters, and decide next steps.
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
            CREATE TABLE IF NOT EXISTS data_contracts (v_contract_code TEXT PRIMARY KEY, v_contract_name TEXT, v_contract_description TEXT, v_source_owner TEXT, v_ingestion_owner TEXT, v_source_system TEXT, v_target_system TEXT);
            CREATE TABLE IF NOT EXISTS etl_pipeline_metadata (v_query_code TEXT PRIMARY KEY, v_query_description TEXT, v_target_table_or_object TEXT, v_source_table_or_object TEXT, v_source_type TEXT, v_target_type TEXT, v_from_clause TEXT, v_where_clause TEXT, v_contract_code TEXT, FOREIGN KEY (v_contract_code) REFERENCES data_contracts(v_contract_code));
            CREATE TABLE IF NOT EXISTS etl_pipeline_dependency (v_query_code TEXT, v_depends_on TEXT, FOREIGN KEY (v_query_code) REFERENCES etl_pipeline_metadata(v_query_code), FOREIGN KEY (v_depends_on) REFERENCES etl_pipeline_metadata(v_query_code));
            CREATE TABLE IF NOT EXISTS business_dictionary (v_business_element_code TEXT PRIMARY KEY, v_business_definition TEXT);
            CREATE TABLE IF NOT EXISTS business_element_mapping (v_data_element_code TEXT PRIMARY KEY, v_data_element_name TEXT, v_table_name TEXT, v_business_element_code TEXT, FOREIGN KEY (v_business_element_code) REFERENCES business_dictionary(v_business_element_code));
            CREATE TABLE IF NOT EXISTS transformation_rules (v_transformation_code TEXT PRIMARY KEY, v_transformation_rules TEXT);
            CREATE TABLE IF NOT EXISTS etl_element_mapping (v_query_code TEXT, v_source_data_element_code TEXT, v_target_data_element_code TEXT, v_transformation_code TEXT, FOREIGN KEY (v_query_code) REFERENCES etl_pipeline_metadata_metadata_metadata_metadata_metadata_metadata_metadata_metadata_metadata(v_query_code), FOREIGN KEY (v_source_data_element_code) REFERENCES business_element_mapping(v_data_element_code), FOREIGN KEY (v_target_data_element_code) REFERENCES business_element_mapping(v_data_element_code), FOREIGN KEY (v_transformation_code) REFERENCES transformation_rules(v_transformation_code));
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
                             'ProductTeam', 'ETLTeam', 'ProductDB', 'AnalyticsDB')])
        cursor.executemany("INSERT OR REPLACE INTO etl_pipeline_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [('Q001', 'Load customer dimension table', 'dim_customer', 'customers', 'table', 'table',
                             'FROM customers c', 'WHERE c.active = 1 AND c.created_date >= CURRENT_DATE - 90', 'C001'),
                            ('Q002', 'Load order facts', 'fact_orders', 'orders o JOIN customers c', 'table', 'table',
                             'FROM orders o JOIN customers c ON o.customer_id = c.customer_id',
                             'WHERE o.order_date >= CURRENT_DATE - 30', 'C002'),
                            ('Q003', 'Aggregate sales data', 'agg_sales', 'fact_orders', 'table', 'table',
                             'FROM fact_orders fo', 'GROUP BY fo.prod_id, DATE_TRUNC(month, fo.order_dt)', 'C002')])
        cursor.executemany("INSERT OR REPLACE INTO etl_pipeline_dependency VALUES (?, ?)",
                           [('Q002', 'Q001'), ('Q003', 'Q002')])
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
        "SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts WHERE v_contract_name LIKE ?",
        (f"%{contract_query_param}%",))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {"success": True, "contract_code": result[0], "contract_name": result[1], "description": result[2]}
    return {"success": False, "error": f"Contract '{contract_name}' not found."}


@tool
def query_pipelines_by_contract(contract_code: str) -> Dict[str, Any]:
    """Gets all ETL pipelines for a contract code."""
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT v_query_code, v_query_description FROM etl_pipeline_metadata WHERE v_contract_code = ?",
                   (contract_code,))
    results = cursor.fetchall()
    conn.close()
    pipelines = [{"query_code": row[0], "description": row[1]} for row in results]
    return {"success": True, "pipelines": pipelines}


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


def lineage_coordinator_agent(state: LineageState):
    """Intelligent coordinator that analyzes natural language requests using LLM."""
    logger.info("---EXECUTING: Intelligent Lineage Coordinator---")

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
    - If query mentions specific element names from available elements -> element_based
    - If query mentions "contract", "pipeline" or contract names -> contract_based
    - If query asks to "trace [element_name]" -> element_based
    - Default traversal is bidirectional unless specified
    """

    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{context}\n\nPlease analyze this request and determine:"
                    f"\n1. Lineage type (contract_based or element_based)\n"
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
                                              "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
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
                                              "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
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

                1. **Element-based lineage** - Trace a specific data field
                   Available elements: {', '.join(element_names[:10])}{'...' if len(element_names) > 10 else ''}

                2. **Contract-based lineage** - Analyze a data pipeline
                   Available contracts: {', '.join(contract_names)}

                Please specify which type of lineage you need and be more specific about what you want to trace.""",
                "query_results": {
                    "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)],
                    "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)]
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
        logger.error(f"Error in coordinator agent: {e}")
        traceback.print_exc()
        return {**state, "error_message": f"Coordinator analysis failed: {str(e)}", "current_step": "error"}


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


# lineage_construction_agent function
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
                # Find by name
                for contract in available_contracts:
                    if contract["name"].lower() == selected_name.lower():
                        selected_contract_name = contract["name"]
                        break

            if selected_contract_name:
                return {
                    **state,
                    "contract_name": selected_contract_name,
                    "lineage_type": LineageType.CONTRACT_BASED.value,
                    "requires_human_approval": False,
                    "human_feedback": None,
                    "current_step": "contract_analysis"
                }

        # If we reach here, the selection was invalid
        logger.warning("Invalid human feedback received")
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": (
                "Invalid selection. Please provide either:\n"
                "• A valid number from the options (e.g., 1, 2, 3)\n"
                "• The exact name of the item you want to select\n\n"
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
    """Determines next step based on current state."""
    current_step = state.get("current_step", "")

    if current_step == "error":
        return "error_handler"
    elif state.get("requires_human_approval"):
        return "human_approval"
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


# --- Main Workflow Definition (REVISED) ---
class LineageOrchestrator:
    """Main orchestrator class for the intelligent lineage analysis workflow."""


    def __init__(self):
        self.db_manager = DatabaseManager()
        self.checkpointer = SqliteSaver.from_conn_string("lineage_checkpoints.db")
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.graph = self._build_workflow()
        # Store the complete workflow state instead of just config
        self.paused_state = None
        self.workflow_config = None

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(LineageState)

        # Add all agent nodes
        workflow.add_node("coordinator", lineage_coordinator_agent)
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
        workflow.add_conditional_edges("contract_analysis", should_continue)
        workflow.add_conditional_edges("element_analysis", element_continue_condition)
        workflow.add_conditional_edges("lineage_construction", should_continue)
        workflow.add_conditional_edges("human_approval", should_continue)
        workflow.add_conditional_edges("finalize_results", should_continue)
        workflow.add_edge("error_handler", END)

        # Compile with interrupt before human_approval
        return workflow.compile(checkpointer=self.checkpointer,
                                interrupt_before=["human_approval"])

    async def execute_lineage_request(self, request: LineageRequest) -> Dict[str, Any]:
        """
        Execute a lineage analysis request.
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


from langchain.memory import ConversationSummaryBufferMemory
from typing import Dict, Any, List
import pickle
from datetime import datetime, timedelta


class EnhancedMemoryManager:
    """Enhanced memory system with context awareness and learning capabilities."""

    def __init__(self, max_token_limit: int = 2000):
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=get_llm(),
            max_token_limit=max_token_limit,
            return_messages=True
        )
        self.execution_history = {}  # Track past executions for learning
        self.user_preferences = {}  # Learn user preferences over time
        self.error_patterns = {}  # Track and learn from errors

    def add_execution_memory(self, request: str, result: Dict[str, Any],
                             execution_time: float, success: bool):
        """Store execution patterns for learning."""
        memory_entry = {
            "request": request,
            "result_summary": self._summarize_result(result),
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now(),
            "patterns": self._extract_patterns(request, result)
        }

        request_hash = hash(request.lower().strip())
        if request_hash not in self.execution_history:
            self.execution_history[request_hash] = []
        self.execution_history[request_hash].append(memory_entry)

    def get_similar_executions(self, current_request: str) -> List[Dict[str, Any]]:
        """Find similar past executions to inform current decision."""
        current_patterns = self._extract_request_patterns(current_request)
        similar_executions = []

        for entries in self.execution_history.values():
            for entry in entries:
                if self._calculate_similarity(current_patterns, entry["patterns"]) > 0.7:
                    similar_executions.append(entry)

        return sorted(similar_executions, key=lambda x: x["timestamp"], reverse=True)[:3]

    def learn_user_preferences(self, user_feedback: Dict[str, Any]):
        """Learn from user feedback and preferences."""
        feedback_type = user_feedback.get("type")
        if feedback_type == "preference":
            self.user_preferences.update(user_feedback.get("preferences", {}))
        elif feedback_type == "correction":
            self._update_error_patterns(user_feedback)

    def _extract_patterns(self, request: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from request-result pairs."""
        return {
            "request_type": "element" if any(word in request.lower()
                                             for word in ["trace", "element", "field"]) else "contract",
            "complexity": result.get("complexity_score", 0),
            "node_count": len(result.get("nodes", [])),
            "edge_count": len(result.get("edges", [])),
            "keywords": [word for word in request.lower().split()
                         if len(word) > 3 and word.isalpha()]
        }

    def get_context_for_agent(self, agent_name: str, current_state: Dict[str, Any]) -> str:
        """Provide relevant context to agents based on memory."""
        context_parts = []

        # Add conversation context
        if hasattr(self.conversation_memory, 'buffer'):
            context_parts.append(f"Recent conversation: {self.conversation_memory.buffer}")

        # Add similar execution context
        current_request = current_state.get("input_parameter", "")
        similar_execs = self.get_similar_executions(current_request)
        if similar_execs:
            context_parts.append(
                f"Similar past executions: {[exec['result_summary'] for exec in similar_execs]}"
            )

        # Add user preferences
        if self.user_preferences:
            context_parts.append(f"User preferences: {self.user_preferences}")

        return "\n".join(context_parts)


from functools import wraps
import time
from typing import Callable, Any
from enum import Enum


class ErrorType(Enum):
    NETWORK_ERROR = "network"
    DATABASE_ERROR = "database"
    LLM_ERROR = "llm"
    VALIDATION_ERROR = "validation"
    TIMEOUT_ERROR = "timeout"


class RecoveryStrategy(Enum):
    RETRY_WITH_BACKOFF = "retry_backoff"
    FALLBACK_METHOD = "fallback"
    SIMPLIFY_REQUEST = "simplify"
    REQUEST_HUMAN_HELP = "human_help"


class IntelligentRecoveryManager:
    """Manages intelligent error recovery and retry logic."""

    def __init__(self):
        self.error_history = {}
        self.recovery_strategies = {
            ErrorType.NETWORK_ERROR: [RecoveryStrategy.RETRY_WITH_BACKOFF],
            ErrorType.DATABASE_ERROR: [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.FALLBACK_METHOD],
            ErrorType.LLM_ERROR: [RecoveryStrategy.SIMPLIFY_REQUEST, RecoveryStrategy.FALLBACK_METHOD],
            ErrorType.VALIDATION_ERROR: [RecoveryStrategy.REQUEST_HUMAN_HELP],
            ErrorType.TIMEOUT_ERROR: [RecoveryStrategy.SIMPLIFY_REQUEST, RecoveryStrategy.RETRY_WITH_BACKOFF]
        }

    def classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorType:
        """Classify error type for appropriate recovery strategy."""
        error_str = str(error).lower()

        if any(term in error_str for term in ["connection", "network", "timeout"]):
            return ErrorType.NETWORK_ERROR
        elif any(term in error_str for term in ["database", "sqlite", "sql"]):
            return ErrorType.DATABASE_ERROR
        elif any(term in error_str for term in ["api", "model", "generation"]):
            return ErrorType.LLM_ERROR
        elif any(term in error_str for term in ["validation", "invalid", "missing"]):
            return ErrorType.VALIDATION_ERROR
        else:
            return ErrorType.TIMEOUT_ERROR

    def intelligent_retry(self, max_attempts: int = 3):
        """Decorator for intelligent retry with learning."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        error_type = self.classify_error(e, kwargs)

                        # Learn from error patterns
                        self._record_error(func.__name__, error_type, attempt)

                        # Get recovery strategy
                        strategies = self.recovery_strategies.get(error_type, [])
                        if not strategies or attempt == max_attempts - 1:
                            break

                        # Apply recovery strategy
                        if RecoveryStrategy.RETRY_WITH_BACKOFF in strategies:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            time.sleep(wait_time)
                        elif RecoveryStrategy.SIMPLIFY_REQUEST in strategies:
                            kwargs = self._simplify_request(kwargs)

                # If all retries failed, return structured error
                return self._create_recovery_response(func.__name__, last_error, *args, **kwargs)

            return wrapper

        return decorator

    def _simplify_request(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify request parameters to increase success chance."""
        # Example simplifications
        if "max_depth" in kwargs:
            kwargs["max_depth"] = min(kwargs.get("max_depth", 5), 3)
        if "query_codes" in kwargs and len(kwargs["query_codes"]) > 10:
            kwargs["query_codes"] = kwargs["query_codes"][:5]  # Limit query codes
        return kwargs

    def _create_recovery_response(self, func_name: str, error: Exception,
                                  *args, **kwargs) -> Dict[str, Any]:
        """Create intelligent recovery response."""
        suggestions = self._generate_recovery_suggestions(func_name, error, kwargs)

        return {
            "success": False,
            "error": str(error),
            "recovery_attempted": True,
            "function": func_name,
            "suggestions": suggestions,
            "fallback_available": self._has_fallback_method(func_name),
            "requires_human_intervention": self._requires_human_help(error)
        }


from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class AgentMessage:
    """Enhanced inter-agent communication message."""
    sender: str
    receiver: str
    message_type: str  # 'request', 'response', 'notification', 'escalation'
    content: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    requires_response: bool = False
    correlation_id: str = None
    timestamp: datetime = None


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for enhanced agent capabilities."""

    def can_handle_request(self, request: Dict[str, Any]) -> bool:
        """Check if agent can handle specific request type."""
        ...

    def estimate_complexity(self, request: Dict[str, Any]) -> int:
        """Estimate processing complexity (1-10 scale)."""
        ...

    def suggest_alternative_agent(self, request: Dict[str, Any]) -> Optional[str]:
        """Suggest alternative agent if current agent can't handle request."""
        ...


class EnhancedCoordinatorAgent:
    """Enhanced coordinator with intelligent routing and load balancing."""

    def __init__(self, memory_manager: EnhancedMemoryManager,
                 recovery_manager: IntelligentRecoveryManager):
        self.memory_manager = memory_manager
        self.recovery_manager = recovery_manager
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.agent_load = {agent: 0 for agent in self.agent_capabilities.keys()}

    def intelligent_route_request(self, state: LineageState) -> str:
        """Enhanced routing with capability matching and load balancing."""
        request_context = {
            "input_parameter": state.get('input_parameter', ''),
            "lineage_type": state.get('lineage_type', ''),
            "complexity_indicators": self._extract_complexity_indicators(state),
            "historical_context": self.memory_manager.get_similar_executions(
                state.get('input_parameter', '')
            )
        }

        # Get agent recommendations based on capabilities
        suitable_agents = []
        for agent_name, capabilities in self.agent_capabilities.items():
            if self._agent_can_handle(agent_name, request_context):
                complexity = self._estimate_request_complexity(request_context)
                load = self.agent_load[agent_name]
                suitability_score = capabilities.get("efficiency", 5) - load + (10 - complexity)
                suitable_agents.append((agent_name, suitability_score))

        # Sort by suitability and return best agent
        suitable_agents.sort(key=lambda x: x[1], reverse=True)

        if suitable_agents:
            selected_agent = suitable_agents[0][0]
            self.agent_load[selected_agent] += 1  # Update load
            return selected_agent

        # Fallback to original logic if no suitable agent found
        return self._original_routing_logic(state)

    def _agent_can_handle(self, agent_name: str, request_context: Dict[str, Any]) -> bool:
        """Check if agent can handle the request based on capabilities."""
        capabilities = self.agent_capabilities.get(agent_name, {})
        request_type = request_context.get("lineage_type", "")

        if agent_name == "contract_analysis" and request_type == "contract_based":
            return True
        elif agent_name == "element_analysis" and request_type == "element_based":
            return True
        elif agent_name == "coordinator":
            return True  # Coordinator can always handle initial routing

        return False


# Replace SqliteSaver import with MemorySaver
from langgraph.checkpoint.memory import MemorySaver
from collections import defaultdict
import time
from typing import Dict, Any, Optional, List
import json
import logging


# Enhanced Memory Management Classes
class ConversationMemoryManager:
    """Enhanced conversation memory with context awareness and learning"""

    def __init__(self, max_conversations: int = 100, max_tokens_per_conversation: int = 4000):
        self.max_conversations = max_conversations
        self.max_tokens_per_conversation = max_tokens_per_conversation
        self.conversations = {}
        self.user_preferences = defaultdict(dict)
        self.query_patterns = defaultdict(int)
        self.success_patterns = defaultdict(list)

    def store_conversation(self, thread_id: str, conversation_data: Dict[str, Any]):
        """Store conversation with metadata"""
        self.conversations[thread_id] = {
            'data': conversation_data,
            'timestamp': time.time(),
            'success': conversation_data.get('success', False),
            'lineage_type': conversation_data.get('lineage_type'),
            'complexity_score': conversation_data.get('complexity_score', 0)
        }

        # Track patterns for learning
        if conversation_data.get('success'):
            query_type = conversation_data.get('lineage_type', 'unknown')
            self.success_patterns[query_type].append({
                'input_parameter': conversation_data.get('input_parameter'),
                'traversal_direction': conversation_data.get('traversal_direction'),
                'complexity_score': conversation_data.get('complexity_score', 0)
            })

        # Cleanup old conversations
        self._cleanup_old_conversations()

    def _cleanup_old_conversations(self):
        """Remove oldest conversations when limit exceeded"""
        if len(self.conversations) > self.max_conversations:
            oldest_threads = sorted(
                self.conversations.keys(),
                key=lambda x: self.conversations[x]['timestamp']
            )[:-self.max_conversations]

            for thread_id in oldest_threads:
                del self.conversations[thread_id]

    def get_similar_queries(self, current_query: str, lineage_type: str) -> List[Dict[str, Any]]:
        """Find similar successful queries for context"""
        similar_queries = []

        for pattern in self.success_patterns.get(lineage_type, []):
            if self._calculate_similarity(current_query, pattern['input_parameter']) > 0.6:
                similar_queries.append(pattern)

        return similar_queries[:3]  # Return top 3 similar queries

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Simple similarity calculation"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0


class AdaptiveAgentSystem:
    """System for agents to adapt based on unexpected outputs"""

    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.recovery_strategies = {
            'element_not_found': self._handle_element_not_found,
            'contract_not_found': self._handle_contract_not_found,
            'empty_mappings': self._handle_empty_mappings,
            'llm_timeout': self._handle_llm_timeout,
            'invalid_human_input': self._handle_invalid_human_input
        }
        self.retry_counts = defaultdict(int)
        self.max_retries = 3

    def detect_failure_pattern(self, state: Dict[str, Any], error_msg: str) -> str:
        """Detect the type of failure pattern"""
        error_lower = error_msg.lower()

        if 'not found' in error_lower and 'element' in error_lower:
            return 'element_not_found'
        elif 'not found' in error_lower and 'contract' in error_lower:
            return 'contract_not_found'
        elif 'empty' in error_lower or 'no mappings' in error_lower:
            return 'empty_mappings'
        elif 'timeout' in error_lower or 'llm' in error_lower:
            return 'llm_timeout'
        elif 'invalid' in error_lower and 'input' in error_lower:
            return 'invalid_human_input'
        else:
            return 'unknown_error'

    def attempt_recovery(self, state: Dict[str, Any], failure_pattern: str) -> Dict[str, Any]:
        """Attempt to recover from failure using adaptive strategies"""
        thread_id = state.get('thread_id', 'default')

        if self.retry_counts[thread_id] >= self.max_retries:
            return {**state, 'recovery_failed': True, 'error_message': 'Max retries exceeded'}

        self.retry_counts[thread_id] += 1

        if failure_pattern in self.recovery_strategies:
            return self.recovery_strategies[failure_pattern](state)
        else:
            return self._handle_unknown_error(state)

    def _handle_element_not_found(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for element not found"""
        # Try fuzzy matching with available elements
        available_elements = get_available_elements.invoke({})

        if available_elements.get('success'):
            elements = available_elements['elements']
            input_param = state.get('input_parameter', '').lower()

            # Find closest matches using partial string matching
            candidates = []
            for elem in elements:
                elem_name_lower = elem['element_name'].lower()
                if any(word in elem_name_lower for word in input_param.split()):
                    candidates.append(elem)

            if candidates:
                return {
                    **state,
                    'requires_human_approval': True,
                    'human_approval_message': f"Element not found exactly, but found similar elements:\n" +
                                              "\n".join([f"{i + 1}. {elem['element_name']}" for i, elem in
                                                         enumerate(candidates)]) +
                                              "\n\nWould you like to select one of these?",
                    'query_results': {'suggested_elements': candidates},
                    'recovery_attempted': True
                }

        return {**state, 'recovery_failed': True}

    def _handle_contract_not_found(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for contract not found"""
        # Similar to element recovery but for contracts
        available_contracts = get_available_contracts.invoke({})

        if available_contracts.get('success'):
            contracts = available_contracts['contracts']
            input_param = state.get('input_parameter', '').lower()

            candidates = []
            for contract in contracts:
                if any(word in contract['contract_name'].lower() for word in input_param.split()):
                    candidates.append(contract)

            if candidates:
                return {
                    **state,
                    'requires_human_approval': True,
                    'human_approval_message': f"Contract not found exactly, but found similar contracts:\n" +
                                              "\n".join([f"{i + 1}. {contract['contract_name']}" for i, contract in
                                                         enumerate(candidates)]) +
                                              "\n\nWould you like to select one of these?",
                    'query_results': {'suggested_contracts': candidates},
                    'recovery_attempted': True
                }

        return {**state, 'recovery_failed': True}

    def _handle_empty_mappings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for empty mappings"""
        # Try broader search or suggest alternative approaches
        return {
            **state,
            'requires_human_approval': True,
            'human_approval_message': "No direct mappings found. Would you like to:\n1. Search with broader criteria\n2. Try a different element or contract\n3. Get available options",
            'query_results': {'recovery_options': ['broader_search', 'different_input', 'show_available']},
            'recovery_attempted': True
        }

    def _handle_llm_timeout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for LLM timeout"""
        # Simplify the request and retry with reduced context
        return {
            **state,
            'llm_simplified_mode': True,
            'recovery_attempted': True,
            'current_step': 'coordinator'  # Restart with simplified mode
        }

    def _handle_invalid_human_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for invalid human input"""
        return {
            **state,
            'requires_human_approval': True,
            'human_approval_message': "I didn't understand your input. Please provide:\n• A number from the options\n• The exact name\n• Or type 'help' for more guidance",
            'recovery_attempted': True
        }

    def _handle_unknown_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic recovery strategy"""
        return {
            **state,
            'requires_human_approval': True,
            'human_approval_message': "An unexpected error occurred. Would you like to:\n1. Try a simpler query\n2. See available options\n3. Start over",
            'query_results': {'recovery_options': ['simplify', 'show_options', 'restart']},
            'recovery_attempted': True
        }


# Enhanced LineageOrchestrator with MemorySaver
class EnhancedLineageOrchestrator:
    """Enhanced orchestrator with MemorySaver and adaptive capabilities"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        # Use MemorySaver instead of SqliteSaver
        self.checkpointer = MemorySaver()

        # Enhanced memory management
        self.conversation_memory = ConversationMemoryManager()
        self.adaptive_system = AdaptiveAgentSystem()

        # Context-aware memory
        self.session_context = {}
        self.global_patterns = defaultdict(int)

        self.graph = self._build_workflow()
        self.paused_state = None
        self.workflow_config = None

    def _build_workflow(self):
        """Build enhanced workflow with adaptive agents"""
        workflow = StateGraph(LineageState)

        # Add enhanced agent nodes
        workflow.add_node("coordinator", self._enhanced_coordinator_agent)
        workflow.add_node("contract_analysis", self._enhanced_contract_analysis_agent)
        workflow.add_node("element_analysis", self._enhanced_element_analysis_agent)
        workflow.add_node("lineage_construction", lineage_construction_agent)
        workflow.add_node("human_approval", self._enhanced_human_approval_agent)
        workflow.add_node("finalize_results", self._enhanced_finalize_results_agent)
        workflow.add_node("error_handler", self._adaptive_error_handler_agent)
        workflow.add_node("recovery_handler", self._recovery_handler_agent)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Enhanced conditional routing
        workflow.add_conditional_edges("coordinator", self._enhanced_should_continue)
        workflow.add_conditional_edges("contract_analysis", self._enhanced_should_continue)
        workflow.add_conditional_edges("element_analysis", self._enhanced_element_continue_condition)
        workflow.add_conditional_edges("lineage_construction", self._enhanced_should_continue)
        workflow.add_conditional_edges("human_approval", self._enhanced_should_continue)
        workflow.add_conditional_edges("finalize_results", self._enhanced_should_continue)
        workflow.add_conditional_edges("error_handler", self._enhanced_should_continue)
        workflow.add_conditional_edges("recovery_handler", self._enhanced_should_continue)

        workflow.add_edge("recovery_handler", END)

        return workflow.compile(checkpointer=self.checkpointer, interrupt_before=["human_approval"])

    def _enhanced_coordinator_agent(self, state: LineageState):
        """Enhanced coordinator with memory and learning"""
        logger.info("---EXECUTING: Enhanced Intelligent Lineage Coordinator---")

        try:
            # Check for similar past queries
            user_query = state.get('input_parameter', '')
            thread_id = state.get('thread_id', 'default')

            # Use conversation memory for context
            similar_queries = self.conversation_memory.get_similar_queries(
                user_query,
                state.get('lineage_type', 'unknown')
            )

            if similar_queries:
                logger.info(f"Found {len(similar_queries)} similar successful queries")
                # Use patterns from successful queries to enhance current analysis
                state['similar_patterns'] = similar_queries

            # Call original coordinator logic but with enhanced context
            result = lineage_coordinator_agent(state)

            # Track query patterns
            if result.get('lineage_type'):
                self.global_patterns[result['lineage_type']] += 1

            return result

        except Exception as e:
            logger.error(f"Enhanced coordinator failed: {e}")
            # Attempt recovery
            failure_pattern = self.adaptive_system.detect_failure_pattern(state, str(e))
            return self.adaptive_system.attempt_recovery(state, failure_pattern)

    def _enhanced_contract_analysis_agent(self, state: LineageState):
        """Enhanced contract analysis with adaptive error handling"""
        try:
            result = contract_analysis_agent(state)

            # Check if result is unexpected (empty or error)
            if (not result.get('query_results', {}).get('pipelines') or
                    result.get('error_message')):
                failure_pattern = self.adaptive_system.detect_failure_pattern(
                    state, result.get('error_message', 'Empty results'))
                return self.adaptive_system.attempt_recovery(result, failure_pattern)

            return result

        except Exception as e:
            logger.error(f"Enhanced contract analysis failed: {e}")
            failure_pattern = self.adaptive_system.detect_failure_pattern(state, str(e))
            return self.adaptive_system.attempt_recovery(state, failure_pattern)

    def _enhanced_element_analysis_agent(self, state: LineageState):
        """Enhanced element analysis with recovery mechanisms"""
        try:
            result = element_analysis_agent(state)

            # Check for empty or insufficient mappings
            mappings = result.get('query_results', {}).get('mappings', [])
            if not mappings:
                failure_pattern = 'empty_mappings'
                return self.adaptive_system.attempt_recovery(result, failure_pattern)

            return result

        except Exception as e:
            logger.error(f"Enhanced element analysis failed: {e}")
            failure_pattern = self.adaptive_system.detect_failure_pattern(state, str(e))
            return self.adaptive_system.attempt_recovery(state, failure_pattern)

    def _enhanced_human_approval_agent(self, state: LineageState):
        """Enhanced human approval with better error handling"""
        try:
            result = human_approval_agent(state)

            # If human input was invalid, try recovery
            if (result.get('requires_human_approval') and
                    'invalid' in result.get('human_approval_message', '').lower()):
                failure_pattern = 'invalid_human_input'
                return self.adaptive_system.attempt_recovery(result, failure_pattern)

            return result

        except Exception as e:
            logger.error(f"Enhanced human approval failed: {e}")
            failure_pattern = self.adaptive_system.detect_failure_pattern(state, str(e))
            return self.adaptive_system.attempt_recovery(state, failure_pattern)

    def _enhanced_finalize_results_agent(self, state: LineageState):
        """Enhanced finalization with conversation storage"""
        result = finalize_results_agent(state)

        # Store successful conversation in memory
        if result.get('final_result'):
            thread_id = state.get('thread_id', 'default')
            conversation_data = {
                'input_parameter': state.get('input_parameter'),
                'lineage_type': state.get('lineage_type'),
                'traversal_direction': state.get('traversal_direction'),
                'complexity_score': state.get('complexity_score'),
                'success': True,
                'final_result': result['final_result']
            }
            self.conversation_memory.store_conversation(thread_id, conversation_data)

        return result

    def _adaptive_error_handler_agent(self, state: LineageState):
        """Adaptive error handler that tries recovery before giving up"""
        logger.info("---EXECUTING: Adaptive Error Handler---")

        error_msg = state.get('error_message', 'Unknown error')

        # If recovery was already attempted and failed, use original error handler
        if state.get('recovery_failed') or state.get('recovery_attempted'):
            return error_handler_agent(state)

        # Attempt adaptive recovery
        failure_pattern = self.adaptive_system.detect_failure_pattern(state, error_msg)
        recovery_result = self.adaptive_system.attempt_recovery(state, failure_pattern)

        # If recovery suggests continuation, route to recovery handler
        if recovery_result.get('recovery_attempted'):
            return {**recovery_result, 'current_step': 'recovery_handler'}
        else:
            return error_handler_agent(state)

    def _recovery_handler_agent(self, state: LineageState):
        """Handles recovery attempts"""
        logger.info("---EXECUTING: Recovery Handler---")

        # If human approval is required, route there
        if state.get('requires_human_approval'):
            return {**state, 'current_step': 'human_approval'}

        # If recovery suggests restarting coordinator
        if state.get('current_step') == 'coordinator':
            return {**state, 'current_step': 'coordinator'}

        # Default to error if no clear recovery path
        return {**state, 'current_step': 'error'}

    def _enhanced_should_continue(self, state: LineageState):
        """Enhanced routing with recovery logic"""
        current_step = state.get("current_step", "")

        # Check for recovery scenarios first
        if state.get('recovery_attempted') and not state.get('recovery_failed'):
            if state.get('requires_human_approval'):
                return "human_approval"
            elif current_step == 'coordinator':
                return "coordinator"
            else:
                return "recovery_handler"

        # Original routing logic
        if current_step == "error":
            return "error_handler"
        elif state.get("requires_human_approval"):
            return "human_approval"
        elif current_step == "contract_analysis":
            return "contract_analysis"
        elif current_step == "element_analysis":
            return "element_analysis"
        elif current_step == "lineage_construction":
            return "lineage_construction"
        elif current_step == "finalize_results":
            return "finalize_results"
        elif current_step == "recovery_handler":
            return "recovery_handler"
        elif current_step == "complete":
            return END
        else:
            return "coordinator"

    def _enhanced_element_continue_condition(self, state: LineageState):
        """Enhanced element continuation with recovery checks"""
        # Check for recovery scenarios
        if state.get('recovery_attempted'):
            return self._enhanced_should_continue(state)

        # Original logic
        if state.get("element_queue") and len(state.get("element_queue", [])) > 0:
            return "element_analysis"
        else:
            return "lineage_construction"

    async def execute_lineage_request(self, request: LineageRequest) -> Dict[str, Any]:
        """Enhanced execution with memory and adaptive features"""
        # Reset for new request
        self.paused_state = None
        self.workflow_config = None

        try:
            # Create thread ID for memory tracking
            thread_id = f"thread_{int(time.time())}_{id(request)}"

            initial_state = {
                "messages": [HumanMessage(content=request.query)],
                "input_parameter": request.query,
                "thread_id": thread_id,
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
                "final_result": None,
                "error_message": None,
                # Enhanced fields
                "recovery_attempted": False,
                "recovery_failed": False,
                "similar_patterns": [],
                "llm_simplified_mode": False
            }

            config = {"configurable": {"thread_id": thread_id}}
            self.workflow_config = config

            # Execute with enhanced error handling
            final_state = None
            async for event in self.graph.astream(initial_state, config):
                logger.info(f"Processing enhanced event: {list(event.keys())}")

                for node_name, state_data in event.items():
                    if isinstance(state_data, tuple) and len(state_data) > 1:
                        state_data = state_data[1] if isinstance(state_data[1], dict) else state_data[0]

                    if not isinstance(state_data, dict):
                        continue

                    final_state = state_data

                    # Enhanced human approval handling
                    if state_data.get("requires_human_approval"):
                        logger.info("---ENHANCED WORKFLOW PAUSED FOR HUMAN INPUT---")
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "message": state_data.get("human_approval_message", "Human input required."),
                            "query_results": state_data.get("query_results", {}),
                            "thread_id": thread_id,
                            "recovery_info": {
                                "recovery_attempted": state_data.get("recovery_attempted", False),
                                "similar_patterns": state_data.get("similar_patterns", [])
                            }
                        }

            # Handle final results
            if final_state and final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return {"error": "Enhanced workflow completed but no final result generated"}

        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {e}")
            return {"success": False, "error": f"Enhanced workflow execution failed: {str(e)}"}
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


# Usage Example
def create_enhanced_orchestrator():
    """Factory function to create enhanced orchestrator"""
    return EnhancedLineageOrchestrator()


# Example usage patterns
async def example_usage():
    orchestrator = create_enhanced_orchestrator()

    # Example 1: Normal request
    request = LineageRequest(query="trace customer_id lineage")
    result = await orchestrator.execute_lineage_request(request)

    if result.get("human_input_required"):
        # Handle human input
        feedback = {"selected_index": 0}  # User selects first option
        final_result = await orchestrator.resume_with_feedback(feedback)
        return final_result

    return result


# --- CLI Interface for Testing ---
async def main():
    """Main function for a truly interactive CLI testing experience."""
    memory_manager = EnhancedMemoryManager()
    recovery_manager = IntelligentRecoveryManager()
    orchestrator = create_enhanced_orchestrator()

    # Add enhanced components
    orchestrator.memory_manager = memory_manager
    orchestrator.recovery_manager = recovery_manager


    print("Intelligent Data Lineage Agent with LLM Integration")
    print("=" * 60)
    print("Enter a query like 'trace customer_id' or 'show lineage for Customer Data Pipeline'")

    while True:
        query = input("\nEnter your lineage query (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            continue

        request = LineageRequest(
            query=query,
            context="Need comprehensive analysis with improvement suggestions",
            preferred_output="graph",
            max_depth=5
        )

        # Execute with enhanced capabilities
        result = await orchestrator.execute_lineage_request(request)
        print("\nProcessing your request...")


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
        print("\n" + "=" * 60)
        print("FINAL LINEAGE ANALYSIS RESULTS")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))


# --- Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())