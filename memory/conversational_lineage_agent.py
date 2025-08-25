import json
import asyncio
import traceback
import os
import re
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple, Annotated, Union, TypedDict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain_anthropic import ChatAnthropic
import anthropic  # Keep for compatibility with other modules

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()


# System prompts with ReAct capabilities
SYSTEM_PROMPT = """You are an intelligent Data Lineage Assistant designed to help users analyze and understand data lineage across their data ecosystem. You have capabilities to:

1. Trace data through contracts, pipelines, and individual elements
2. Analyze transformation logic and understand business rules
3. Identify data quality and governance issues
4. Provide insights about data flow patterns
5. Create visualizations of data lineage paths

To achieve these goals, you can use various tools. ALWAYS analyze what the user is asking for and THINK about which tools you need to use to answer their question completely. The tools can access a database with information about data contracts, pipelines, elements, transformations, and their relationships.

When analyzing a request:
1. Determine if the user is asking about a specific contract, pipeline, or data element
2. Consider if they want upstream, downstream, or bidirectional lineage
3. Use the appropriate tools to gather the required information
4. Present the results in a clear, conversational way
5. Make sure to respond to follow-up questions by considering the previous conversation context

CRITICAL - METADATA LIMITATIONS:
- You MUST ONLY provide information that is explicitly available in the metadata accessed through your tools
- NEVER make up or hallucinate information that isn't found in the metadata
- If a user asks about a contract, pipeline, element, or relationship that doesn't exist in the metadata, CLEARLY state that it's not in the available metadata
- DO NOT attempt to fill in gaps with speculative information when data is missing
- It's much better to say "I don't have information about that" than to make up an answer
- If you suspect a user might be asking about something similar to what's in the metadata, suggest specific alternatives that ARE in the metadata

CRITICAL - EXPRESSING UNCERTAINTY:
- If you don't have enough information to provide a complete or accurate answer, you MUST express your uncertainty clearly
- Use phrases like "I'm not sure", "I don't have enough information", "I need additional details"
- When uncertain, explicitly ask clarifying questions to gather the information you need
- NEVER make up information or provide speculative answers when uncertain
- It's better to admit uncertainty than to provide potentially incorrect information
- Consider explicitly stating what additional information would help you provide a better answer

Remember that the user may ask follow-up questions about previous queries, so maintain context and refer back to previously retrieved information when relevant.

IMPORTANT: Follow the ReAct format:
1. Thought: Analyze the task, consider what information you need
2. Action: Use an available tool
3. Action Input: Provide the necessary parameters to the tool
4. Observation: Review the tool's output
5. ... repeat steps 1-4 as needed ...
6. Thought: Determine the final response
7. Answer: Provide a clear, informative response to the user OR express uncertainty and ask clarifying questions

Available tools: {tools}
"""

class ConversationState(TypedDict):
    """State that is maintained for the conversation."""
    messages: List[BaseMessage]
    context: Dict[str, Any]


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
            CREATE TABLE IF NOT EXISTS etl_element_mapping (v_query_code TEXT, v_source_data_element_code TEXT, v_target_data_element_code TEXT, v_transformation_code TEXT, FOREIGN KEY (v_query_code) REFERENCES etl_pipeline_metadata(v_query_code), FOREIGN KEY (v_source_data_element_code) REFERENCES business_element_mapping(v_data_element_code), FOREIGN KEY (v_target_data_element_code) REFERENCES business_element_mapping(v_data_element_code), FOREIGN KEY (v_transformation_code) REFERENCES transformation_rules(v_transformation_code));
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
                            ('DE008', 'customer_id', 'dim_customer', 'BE001'),
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


# --- Tool Definitions ---

@tool()
def query_contract_by_name(contract_name: str) -> Dict[str, Any]:
    """
    Queries data contract table by contract name.
    This tool is useful when a user asks about a specific contract or mentions a contract by name.
    
    Args:
        contract_name: The name of the contract to search for, can be partial name.
    
    Returns:
        Dict with contract details if found, or error message if not found.
    """
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
    """
    Gets all ETL pipelines for a contract code.
    Use this to find all the data pipelines associated with a particular contract.
    
    Args:
        contract_code: The unique code of the contract (e.g., C001)
    
    Returns:
        Dict with a list of pipelines belonging to the contract.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT v_query_code, v_query_description, v_target_table_or_object, v_source_table_or_object 
    FROM etl_pipeline_metadata 
    WHERE v_contract_code = ?
    """, (contract_code,))
    results = cursor.fetchall()
    conn.close()
    pipelines = [{"query_code": row[0], "description": row[1], "target": row[2], "source": row[3]} for row in results]
    return {"success": True, "pipelines": pipelines}


@tool
def query_pipeline_dependencies(query_codes: List[str]) -> Dict[str, Any]:
    """
    Gets downstream pipeline dependencies for a given list of query codes.
    This helps understand how different pipelines depend on each other.
    
    Args:
        query_codes: List of query codes (e.g., ["Q001", "Q002"])
    
    Returns:
        Dict with dependencies between pipelines.
    """
    if not query_codes: 
        return {"success": True, "dependencies": {}}
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in query_codes])
    cursor.execute(
        f"SELECT v_query_code, v_depends_on FROM etl_pipeline_dependency WHERE v_query_code IN ({placeholders}) OR v_depends_on IN ({placeholders})",
        query_codes + query_codes)
    results = cursor.fetchall()
    conn.close()
    dependencies = {}
    for from_q, to_q in results:
        if from_q not in dependencies: 
            dependencies[from_q] = []
        dependencies[from_q].append(to_q)
    return {"success": True, "dependencies": dependencies}


@tool
def query_element_mappings_by_queries(query_codes: List[str]) -> Dict[str, Any]:
    """
    Gets element mappings for specific query codes.
    This shows how data elements are transformed from source to target within specified ETL pipelines.
    
    Args:
        query_codes: List of query codes (e.g., ["Q001", "Q002"])
    
    Returns:
        Dict with element mappings information.
    """
    if not query_codes: 
        return {"success": True, "mappings": []}
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in query_codes])
    query = f"""
        SELECT eem.v_query_code, eem.v_source_data_element_code, eem.v_target_data_element_code, tr.v_transformation_rules,
               src.v_data_element_name, src.v_table_name, tgt.v_data_element_name, tgt.v_table_name
        FROM etl_element_mapping eem
        left outer JOIN business_element_mapping src ON eem.v_source_data_element_code = src.v_data_element_code
        left outer JOIN business_element_mapping tgt ON eem.v_target_data_element_code = tgt.v_data_element_code
        left outer JOIN transformation_rules tr ON eem.v_transformation_code = tr.v_transformation_code
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
    """
    Finds a data element by its name.
    Use this when a user asks about a specific data element or column.
    
    Args:
        element_name: The name of the data element to search for (e.g., "customer_id")
    
    Returns:
        Dict with matching elements information.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT bem.v_data_element_code, bem.v_data_element_name, bem.v_table_name, bd.v_business_definition FROM business_element_mapping bem LEFT JOIN business_dictionary bd ON bem.v_business_element_code = bd.v_business_element_code WHERE bem.v_data_element_name LIKE ?",
        (f"%{element_name}%",))
    results = cursor.fetchall()
    conn.close()
    if results:
        elements = [{"element_code": r[0], "element_name": r[1], "table_name": r[2], "business_definition": r[3]} for r in results]
        return {"success": True, "elements": elements}
    return {"success": False, "error": f"Element '{element_name}' not found."}


@tool
def trace_element_connections(element_code: str, direction: str) -> Dict[str, Any]:
    """
    Traces connections for a data element in the specified direction.
    This helps understand how data flows to or from a particular element.
    
    Args:
        element_code: The code of the data element (e.g., "DE001")
        direction: The direction to trace ("upstream", "downstream", or "bidirectional")
    
    Returns:
        Dict with connections information.
    """
    connections = []
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()

    if direction in ['downstream', 'bidirectional']:
        cursor.execute("""
            SELECT eem.v_target_data_element_code, bem.v_data_element_name, bem.v_table_name, eem.v_query_code, tr.v_transformation_rules
            FROM etl_element_mapping eem
            JOIN business_element_mapping bem ON eem.v_target_data_element_code = bem.v_data_element_code
            LEFT JOIN transformation_rules tr ON eem.v_transformation_code = tr.v_transformation_code
            WHERE eem.v_source_data_element_code = ?
        """, (element_code,))
        connections.extend([{
            "connected_code": r[0], 
            "connected_name": r[1],
            "connected_table": r[2],
            "query_code": r[3],
            "transformation": r[4],
            "direction": "downstream"
        } for r in cursor.fetchall()])
    
    if direction in ['upstream', 'bidirectional']:
        cursor.execute("""
            SELECT eem.v_source_data_element_code, bem.v_data_element_name, bem.v_table_name, eem.v_query_code, tr.v_transformation_rules
            FROM etl_element_mapping eem
            JOIN business_element_mapping bem ON eem.v_source_data_element_code = bem.v_data_element_code
            LEFT JOIN transformation_rules tr ON eem.v_transformation_code = tr.v_transformation_code
            WHERE eem.v_target_data_element_code = ?
        """, (element_code,))
        connections.extend([{
            "connected_code": r[0], 
            "connected_name": r[1],
            "connected_table": r[2],
            "query_code": r[3],
            "transformation": r[4],
            "direction": "upstream"
        } for r in cursor.fetchall()])

    conn.close()
    return {"success": True, "connections": connections}


@tool
def get_element_details(element_code: str) -> Dict[str, Any]:
    """
    Gets detailed information about a specific data element by its code.
    
    Args:
        element_code: The code of the data element (e.g., "DE001")
    
    Returns:
        Dict with element details including business definition.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT bem.v_data_element_code, bem.v_data_element_name, bem.v_table_name, 
               bem.v_business_element_code, bd.v_business_definition
        FROM business_element_mapping bem
        LEFT JOIN business_dictionary bd ON bem.v_business_element_code = bd.v_business_element_code
        WHERE bem.v_data_element_code = ?
    """, (element_code,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "success": True,
            "element_code": result[0],
            "element_name": result[1],
            "table_name": result[2],
            "business_element_code": result[3],
            "business_definition": result[4]
        }
    return {"success": False, "error": f"Element code '{element_code}' not found."}


@tool
def get_all_query_codes() -> Dict[str, Any]:
    """
    Dynamically fetch all available query codes from database.
    This is useful when you need to know all ETL processes in the system.
    
    Returns:
        Dict with list of all query codes.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT v_query_code, v_query_description, v_target_table_or_object, v_source_table_or_object 
        FROM etl_pipeline_metadata 
        ORDER BY v_query_code
    """)
    results = cursor.fetchall()
    conn.close()

    query_details = [{
        "query_code": row[0], 
        "description": row[1], 
        "target": row[2], 
        "source": row[3]
    } for row in results]
    
    return {"success": True, "queries": query_details}


@tool
def get_available_contracts() -> Dict[str, Any]:
    """
    Dynamically fetch all available contracts from database.
    Use this when you need to list all data contracts in the system.
    
    Returns:
        Dict with list of all contracts.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts")
    results = cursor.fetchall()
    conn.close()

    contracts = [{"contract_code": r[0], "contract_name": r[1], "description": r[2]} for r in results]
    return {"success": True, "contracts": contracts}


@tool
def get_available_elements() -> Dict[str, Any]:
    """
    Dynamically fetch all available data elements from database.
    This helps when you need to know all data elements in the system.
    
    Returns:
        Dict with list of all data elements.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT bem.v_data_element_code, bem.v_data_element_name, bem.v_table_name, bd.v_business_definition FROM business_element_mapping bem LEFT JOIN business_dictionary bd ON bem.v_business_element_code = bd.v_business_element_code ORDER BY bem.v_data_element_name")
    results = cursor.fetchall()
    conn.close()

    elements = [{
        "element_code": r[0],
        "element_name": r[1], 
        "table_name": r[2],
        "business_definition": r[3]
    } for r in results]
    
    return {"success": True, "elements": elements}


@tool
def get_query_details(query_code: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific ETL query.
    
    Args:
        query_code: The code of the query (e.g., "Q001")
    
    Returns:
        Dict with query details.
    """
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v_query_code, v_query_description, v_target_table_or_object, v_source_table_or_object, 
               v_source_type, v_target_type, v_from_clause, v_where_clause, v_contract_code
        FROM etl_pipeline_metadata
        WHERE v_query_code = ?
    """, (query_code,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return {"success": False, "error": f"Query code '{query_code}' not found."}
    
    # Get contract details
    contract_details = {}
    if result[8]:
        conn = db_manager_global.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT v_contract_name, v_contract_description FROM data_contracts WHERE v_contract_code = ?", 
            (result[8],))
        contract_result = cursor.fetchone()
        conn.close()
        
        if contract_result:
            contract_details = {
                "contract_code": result[8],
                "contract_name": contract_result[0],
                "contract_description": contract_result[1]
            }
    
    # Get element mappings
    element_mappings = query_element_mappings_by_queries({"query_codes": [query_code]})
    
    return {
        "success": True,
        "query_details": {
            "query_code": result[0],
            "description": result[1],
            "target_object": result[2],
            "source_object": result[3],
            "source_type": result[4],
            "target_type": result[5],
            "from_clause": result[6],
            "where_clause": result[7],
            "contract": contract_details
        },
        "element_mappings": element_mappings.get("mappings", [])
    }


@tool
def get_full_lineage_path(element_code: str, direction: str, max_depth: int = 5) -> Dict[str, Any]:
    """
    Get the complete lineage path for an element, recursively tracing connections up to max_depth.
    
    Args:
        element_code: The code of the data element (e.g., "DE001")
        direction: The direction to trace ("upstream", "downstream", "bidirectional")
        max_depth: Maximum recursion depth (default: 5)
    
    Returns:
        Dict with complete lineage information including all nodes and edges.
    """
    def trace_recursive(code, direction, current_depth, visited):
        if current_depth > max_depth or code in visited:
            return [], []
        
        visited.add(code)
        
        # Get element details
        element_details = get_element_details({"element_code": code})
        if not element_details.get("success"):
            return [], []
        
        # Create node for this element
        node = {
            "id": code,
            "name": element_details["element_name"],
            "table": element_details["table_name"],
            "business_definition": element_details["business_definition"]
        }
        
        nodes = [node]
        edges = []
        
        # Get connections
        connections = trace_element_connections({"element_code": code, "direction": direction})
        
        for conn in connections.get("connections", []):
            conn_code = conn["connected_code"]
            # Recursively get nodes and edges for the connected element
            if conn["direction"] == "downstream":
                child_nodes, child_edges = trace_recursive(conn_code, direction, current_depth + 1, visited)
                # Add edge from this element to child
                edges.append({
                    "source": code,
                    "target": conn_code,
                    "query": conn["query_code"],
                    "transformation": conn["transformation"]
                })
            else:  # upstream
                child_nodes, child_edges = trace_recursive(conn_code, direction, current_depth + 1, visited)
                # Add edge from parent to this element
                edges.append({
                    "source": conn_code,
                    "target": code,
                    "query": conn["query_code"],
                    "transformation": conn["transformation"]
                })
            
            nodes.extend(child_nodes)
            edges.extend(child_edges)
        
        return nodes, edges

    visited = set()
    nodes, edges = trace_recursive(element_code, direction, 0, visited)
    
    # Deduplicate nodes
    unique_nodes = {}
    for node in nodes:
        if node["id"] not in unique_nodes:
            unique_nodes[node["id"]] = node
    
    # Deduplicate edges
    edge_keys = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["query"])
        if key not in edge_keys:
            unique_edges.append(edge)
            edge_keys.add(key)
    
    return {
        "success": True,
        "nodes": list(unique_nodes.values()),
        "edges": unique_edges,
        "element_count": len(unique_nodes),
        "connection_count": len(unique_edges)
    }


@tool
def search_lineage_database(search_term: str) -> Dict[str, Any]:
    """
    Search across all database tables for the given search term.
    This is useful for general searches when you're not sure where to look.
    
    Args:
        search_term: Term to search for in the database
    
    Returns:
        Dict with search results from various tables.
    """
    results = {
        "contracts": [],
        "pipelines": [],
        "elements": [],
        "transformations": []
    }
    
    conn = db_manager_global.get_connection()
    cursor = conn.cursor()
    
    # Search in contracts
    cursor.execute(
        "SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts WHERE v_contract_name LIKE ? OR v_contract_description LIKE ?", 
        (f"%{search_term}%", f"%{search_term}%"))
    results["contracts"] = [{"code": r[0], "name": r[1], "description": r[2]} for r in cursor.fetchall()]
    
    # Search in pipelines
    cursor.execute(
        "SELECT v_query_code, v_query_description, v_target_table_or_object, v_source_table_or_object FROM etl_pipeline_metadata WHERE v_query_description LIKE ? OR v_target_table_or_object LIKE ? OR v_source_table_or_object LIKE ?", 
        (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
    results["pipelines"] = [{"code": r[0], "description": r[1], "target": r[2], "source": r[3]} for r in cursor.fetchall()]
    
    # Search in elements
    cursor.execute(
        "SELECT bem.v_data_element_code, bem.v_data_element_name, bem.v_table_name, bd.v_business_definition FROM business_element_mapping bem LEFT JOIN business_dictionary bd ON bem.v_business_element_code = bd.v_business_element_code WHERE bem.v_data_element_name LIKE ? OR bem.v_table_name LIKE ? OR bd.v_business_definition LIKE ?", 
        (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
    results["elements"] = [{"code": r[0], "name": r[1], "table": r[2], "definition": r[3]} for r in cursor.fetchall()]
    
    # Search in transformations
    cursor.execute(
        "SELECT v_transformation_code, v_transformation_rules FROM transformation_rules WHERE v_transformation_rules LIKE ?", 
        (f"%{search_term}%",))
    results["transformations"] = [{"code": r[0], "rules": r[1]} for r in cursor.fetchall()]
    
    conn.close()
    
    # Count total results
    total_results = sum(len(v) for v in results.values())
    
    return {
        "success": True,
        "search_term": search_term,
        "total_results": total_results,
        "results": results
    }


def get_llm():
    """Helper function to get the LLM instance."""
    # Use the LangChain Anthropic Chat model which adapts the API to have .invoke() method
    # This makes it compatible with the LangChain interface
    if "ANTHROPIC_API_KEY" in os.environ:
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        return ChatAnthropic(model=model_name, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("No valid model found!!")
        #     if "ANTHROPIC_API_KEY" in os.environ:
        # return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


# Helper function to list available conversation sessions
def list_conversation_sessions(memory_db_path="lineage_memory.db"):
    """
    List all available conversation sessions in the SQLite database.
    
    Args:
        memory_db_path: Path to the SQLite database
        
    Returns:
        List of tuples containing (session_id, timestamp, message_count, last_query)
        Each tuple contains the session ID, its creation timestamp, number of messages, 
        and the last user query in that session
    """
    if not os.path.exists(memory_db_path):
        logger.info(f"Memory database file not found: {memory_db_path}")
        return []
    
    try:
        # Connect to the database
        conn = sqlite3.connect(memory_db_path)
        cursor = conn.cursor()
        
        # Check if the langchain_sqlite_message_store table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='langchain_sqlite_message_store'")
        if not cursor.fetchone():
            logger.info(f"No conversation history table found in database: {memory_db_path}")
            conn.close()
            return []
            
        # Get session information - ID, first message time, message count, and last query
        cursor.execute("""
            SELECT 
                session_id, 
                MIN(create_time) as first_message_time,
                COUNT(*) as message_count,
                (
                    SELECT content 
                    FROM langchain_sqlite_message_store 
                    WHERE session_id = s.session_id 
                    AND type = 'human' 
                    ORDER BY create_time DESC 
                    LIMIT 1
                ) as last_query
            FROM langchain_sqlite_message_store s
            GROUP BY session_id
            ORDER BY first_message_time DESC
        """)
        
        # Process the results
        sessions = []
        for row in cursor.fetchall():
            session_id = row[0]
            timestamp = datetime.fromtimestamp(row[1]).strftime('%Y-%m-%d %H:%M:%S')
            message_count = row[2]
            last_query = row[3]
            
            # Truncate long queries for display
            if last_query and len(last_query) > 50:
                last_query = last_query[:47] + "..."
                
            sessions.append((session_id, timestamp, message_count, last_query))
        
        conn.close()
        return sessions
    except Exception as e:
        logger.error(f"Error listing conversation sessions: {e}")
        return []

# Helper function to clean ReAct output format
def clean_react_output(text: str) -> str:
    """
    Clean the ReAct format output by removing thought process and tool interactions.
    
    Args:
        text: The raw output from the LLM
        
    Returns:
        Cleaned text with only the final answer
    """
    # Extract the final answer if it exists
    if "Answer:" in text:
        parts = text.split("Answer:", 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    # If no "Answer:" tag is found, try to extract content after the last Thought/Action/Observation block
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Thought:") or lines[i].startswith("Action:") or lines[i].startswith("Observation:"):
            if i + 1 < len(lines):
                return "\n".join(lines[i+1:]).strip()
    
    # If nothing else works, return the original text but strip any leading Thought: marker
    if text.startswith("Thought:"):
        return text.replace("Thought:", "", 1).strip()
    
    return text.strip()

# --- LangGraph setup for Conversational Agent with ReAct ---
def create_agent():
    """Create the conversational agent with ReAct capabilities."""
    llm = get_llm()
    
    # Define all tools
    tools = [
        query_contract_by_name,
        query_pipelines_by_contract,
        query_pipeline_dependencies,
        query_element_mappings_by_queries,
        find_element_by_name,
        trace_element_connections,
        get_element_details,
        get_all_query_codes,
        get_available_contracts,
        get_available_elements,
        get_query_details,
        get_full_lineage_path,
        search_lineage_database,
    ]
    
    # Create the system prompt with tools
    tool_descriptions = []
    for t in tools:
        if hasattr(t, 'name') and hasattr(t, 'description'):
            tool_descriptions.append(f"{t.name}: {t.description}")
        else:
            # Fallback for tools without proper attributes
            tool_descriptions.append(f"{t.__name__}: A data lineage tool")
    
    system_prompt = SYSTEM_PROMPT.format(
        tools="\n".join(tool_descriptions)
    )
    
    # Define the agent state
    class AgentState(TypedDict):
        messages: List[BaseMessage]
        
    # Function to add user message to the state
    def add_user_message(state: AgentState, message: str) -> AgentState:
        """Add a user message to the state, preserving conversation history."""
        return {
            "messages": [*state["messages"], HumanMessage(content=message)]
        }
    
    # Function to manually execute a tool by name and args
    def execute_tool(tool_name: str, **kwargs):
        """Execute a tool by its name with the given arguments."""
        for tool in tools:
            if tool.name == tool_name:
                return tool.invoke(kwargs)
        raise ValueError(f"Tool {tool_name} not found")
    
    # Agent for handling the interaction
    def agent(state: AgentState) -> AgentState:
        """Process the messages and decide on a response or tool use."""
        try:
            # Check if we already have a system message
            has_system_message = any(isinstance(msg, SystemMessage) for msg in state["messages"])
            
            # Get all messages for context
            if has_system_message:
                all_messages = state["messages"]
            else:
                all_messages = [SystemMessage(content=system_prompt)] + state["messages"]
            
            # Get response from LLM
            response = llm.invoke(all_messages)
            content = response.content
            
            # Clean up the output by removing ReAct format markers
            clean_content = clean_react_output(content)
            
            # Add agent's response to messages
            state["messages"].append(AIMessage(content=clean_content))
            return state
        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            state["messages"].append(AIMessage(content=f"I encountered an error: {str(e)}"))
            return state
    
    # Define a simpler graph without multiple nodes
    def run_agent(state, message):
        """Run the full agent cycle with the user message."""
        # If we already added the message (in process_message), don't add it again
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage) and state["messages"][-1].content == message:
            new_state = state
        else:
            # Add user message to state
            new_state = add_user_message(state, message)
            
        # Enhance with memory context if available
        if "memory" in state and state["memory"]:
            # Create a memory context message to help the agent understand context
            memory_prompt = _create_memory_context_message(state["memory"])
            if memory_prompt:
                # Check if there's already a system message
                has_system_message = any(isinstance(msg, SystemMessage) for msg in new_state["messages"])
                
                if not has_system_message:
                    # If no system message exists, add it at the beginning
                    new_state["messages"].insert(0, SystemMessage(content=system_prompt + "\n\n" + memory_prompt))
                else:
                    # Update the first system message with memory context
                    for i, msg in enumerate(new_state["messages"]):
                        if isinstance(msg, SystemMessage):
                            # Append memory context to existing system message
                            new_state["messages"][i] = SystemMessage(content=msg.content + "\n\n" + memory_prompt)
                            break
        
        # Process with the agent
        return agent(new_state)
        
    def _create_memory_context_message(memory_context):
        """Create a memory context message to guide the agent."""
        if not memory_context:
            return None
            
        context_parts = []
        
        # Add metadata matching information if available
        if "metadata_context" in memory_context:
            metadata_context = memory_context["metadata_context"]
            if not metadata_context.get("has_matching_metadata", True):
                context_parts.append("⚠️ WARNING: The user's query doesn't match any specific metadata in our database. " +
                                    "Be EXTREMELY careful not to hallucinate information. " +
                                    "Only provide information that is explicitly found in the metadata or indicate that " +
                                    "the requested information is not available.")
        
        # Add conversation history summary if available
        if "conversation_history" in memory_context and memory_context["conversation_history"]:
            history_data = memory_context["conversation_history"]
            conversation_history = []
            
            # Handle different formats of history
            if "history" in history_data:
                conversation_history = history_data.get("history", [])
            elif "messages" in history_data:
                messages = history_data["messages"]
                # Extract message pairs
                for i in range(0, len(messages)-1, 2):
                    if i+1 < len(messages):
                        user_msg = messages[i].content if hasattr(messages[i], "content") else str(messages[i])
                        ai_msg = messages[i+1].content if hasattr(messages[i+1], "content") else str(messages[i+1])
                        conversation_history.append({"input": user_msg, "output": ai_msg})
            
            if conversation_history:
                context_parts.append("Previous conversation context:")
                for idx, exchange in enumerate(conversation_history):
                    if isinstance(exchange, dict) and "input" in exchange and "output" in exchange:
                        context_parts.append(f"User: {exchange['input']}")
                        if exchange['output']:
                            context_parts.append(f"Assistant: {exchange['output']}")
                    elif isinstance(exchange, tuple) and len(exchange) == 2:
                        context_parts.append(f"User: {exchange[0]}")
                        if exchange[1]:
                            context_parts.append(f"Assistant: {exchange[1]}")
                    elif idx > 0:  # Skip the first item which is usually just formatting info
                        context_parts.append(str(exchange))
            
        # Add entity information if available
        if "entities" in memory_context:
            entities = memory_context["entities"]
            entity_parts = []
            
            if "contract" in entities:
                entity_parts.append(f"Contract: {entities['contract']}")
                
            if "element" in entities:
                entity_parts.append(f"Data Element: {entities['element']}")
                
            if "query_code" in entities:
                entity_parts.append(f"Query Code: {entities['query_code']}")
                
            if "direction" in entities:
                entity_parts.append(f"Direction: {entities['direction']}")
                
            if entity_parts:
                context_parts.append("Previously discussed entities:\n- " + "\n- ".join(entity_parts))
                
        # Add reference flag if this is likely a follow-up question
        if memory_context.get("references_previous", False):
            context_parts.append(
                "This appears to be a follow-up question that may reference previously mentioned entities."
            )
            
        if context_parts:
            return "MEMORY CONTEXT:\n" + "\n\n".join(context_parts)
    
    # Simplified approach - just return the function
    return run_agent
    
    # Compile the graph
    return workflow.compile()


class ConversationalLineageAgent:
    """Main class for the conversational lineage agent."""
    
    def __init__(self, memory_type="buffer", memory_db_path="lineage_memory.db", memory_k=5, 
                 interface_mode="cli", enable_human_feedback=True):
        """
        Initialize the agent with conversation memory.
        
        Args:
            memory_type: Type of memory to use ("buffer" for all history, "window" for recent k turns)
            memory_db_path: Path to SQLite DB for persistent memory (defaults to "lineage_memory.db")
                           If None, in-memory storage will be used
            memory_k: Number of conversation turns to remember when using window memory
            interface_mode: The interface being used ("cli" or "web")
            enable_human_feedback: Whether to enable human-in-the-loop feedback for uncertain responses
        """
        logger.info("Initializing ConversationalLineageAgent")
        
        # Set up memory
        self.memory_type = memory_type
        self.session_id = f"lineage_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.interface_mode = interface_mode
        self.enable_human_feedback = enable_human_feedback
        
        if memory_db_path is None:
            # In-memory storage
            logger.info("Using in-memory conversation storage (conversations will not persist)")
            if memory_type == "window":
                self.memory = ConversationBufferWindowMemory(
                    k=memory_k,
                    return_messages=True
                )
            else:  # buffer
                self.memory = ConversationBufferMemory(
                    return_messages=True
                )
        else:
            # Create directory for the database if it doesn't exist
            db_dir = os.path.dirname(os.path.abspath(memory_db_path))
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir)
                    logger.info(f"Created directory for memory database: {db_dir}")
                except Exception as e:
                    logger.warning(f"Could not create directory for memory database: {e}")
            
            # Use persistent memory with SQLite backend
            logger.info(f"Using persistent memory with database at {memory_db_path}")
            message_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=f"sqlite:///{memory_db_path}"
            )
            
            # Create the appropriate memory type
            if memory_type == "window":
                self.memory = ConversationBufferWindowMemory(
                    chat_memory=message_history,
                    k=memory_k,
                    return_messages=True
                )
            else:  # buffer
                self.memory = ConversationBufferMemory(
                    chat_memory=message_history,
                    return_messages=True
                )
        
        # Initialize agent and conversation state
        self.agent = create_agent()
        self.conversation_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT)
            ],
            "memory": {}  # For storing extracted entities and context
        }
        
        logger.info("ConversationalLineageAgent initialized successfully")
        
    def _get_css_styles(self) -> str:
        """Return CSS styles for formatted output."""
        return """
<style>
/* Table Styles */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #4285f4;
    color: white;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
tr:hover {
    background-color: #e6f2ff;
}

/* Lineage Graph Styles */
.lineage-graph {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #f9f9f9;
}
.graph-legend {
    display: flex;
    margin-bottom: 15px;
}
.legend-item {
    margin-right: 20px;
    display: flex;
    align-items: center;
}
.node-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #4285f4;
    margin-right: 5px;
}
.edge-line {
    display: inline-block;
    width: 20px;
    height: 2px;
    background-color: #34a853;
    margin-right: 5px;
}
.graph-container {
    background-color: white;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
}
.graph-instructions {
    margin-top: 10px;
    font-size: 0.9em;
    color: #666;
    font-style: italic;
}
</style>
        """

    def _get_css_styles(self) -> str:
        """
        Define CSS styles for tables and graph visualizations.
        
        Returns:
            CSS styles as a string
        """
        return """
        <style>
            /* Table Styles */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
                font-size: 0.9em;
                font-family: sans-serif;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                border-radius: 5px;
                overflow: hidden;
            }
            
            table thead tr {
                background-color: #4285f4;
                color: #ffffff;
                text-align: left;
            }
            
            table th,
            table td {
                padding: 12px 15px;
                border-bottom: 1px solid #dddddd;
            }
            
            table tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            
            table tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            
            table tbody tr:last-of-type {
                border-bottom: 2px solid #4285f4;
            }
            
            /* Lineage Graph Styles */
            .lineage-graph {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
                overflow: hidden;
            }
            
            .lineage-graph svg {
                display: block;
                margin: 0 auto;
            }
            
            .node circle {
                fill: #4285f4;
                stroke: #ffffff;
                stroke-width: 2px;
            }
            
            .node text {
                font-family: sans-serif;
                font-size: 12px;
            }
            
            .link {
                fill: none;
                stroke: #999;
                stroke-opacity: 0.6;
                stroke-width: 2px;
            }
            
            .arrowhead {
                fill: #999;
            }
            
            .node-source circle {
                fill: #34a853; /* Green for source nodes */
            }
            
            .node-target circle {
                fill: #ea4335; /* Red for target nodes */
            }
            
            .node-transform circle {
                fill: #fbbc05; /* Yellow for transformation nodes */
            }
        </style>
        """
    
    def _format_json_as_table(self, data: dict) -> str:
        """
        Format a JSON dictionary as an HTML table.
        
        Args:
            data: Dictionary to format
            
        Returns:
            HTML table representation
        """
        if not data:
            return "<em>Empty data</em>"
        
        html = "<table>\n<thead>\n<tr>\n"
        html += "<th>Property</th><th>Value</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"
        
        for key, value in data.items():
            # Format the value based on its type
            if isinstance(value, (dict, list)):
                formatted_value = f"<pre>{json.dumps(value, indent=2)}</pre>"
            elif isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            elif value is None:
                formatted_value = "<em>None</em>"
            else:
                formatted_value = str(value)
            
            html += f"<tr>\n<td><strong>{key}</strong></td><td>{formatted_value}</td>\n</tr>\n"
        
        html += "</tbody>\n</table>"
        return html
    
    def _format_list_as_table(self, data: list) -> str:
        """
        Format a list of dictionaries as an HTML table.
        
        Args:
            data: List of dictionaries to format
            
        Returns:
            HTML table representation
        """
        if not data:
            return "<em>Empty list</em>"
        
        # If this is not a list of dicts, format differently
        if not all(isinstance(item, dict) for item in data):
            return "<ul>\n" + "\n".join([f"<li>{item}</li>" for item in data]) + "\n</ul>"
        
        # Get all unique keys from all dictionaries
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Sort keys for consistent display
        keys = sorted(all_keys)
        
        # Build the HTML table
        html = "<table>\n<thead>\n<tr>\n"
        for key in keys:
            html += f"<th>{key}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"
        
        for item in data:
            html += "<tr>\n"
            for key in keys:
                value = item.get(key, "")
                # Format the value based on its type
                if isinstance(value, (dict, list)):
                    formatted_value = f"<pre>{json.dumps(value, indent=2)}</pre>"
                elif isinstance(value, bool):
                    formatted_value = "Yes" if value else "No"
                elif value is None:
                    formatted_value = "<em>None</em>"
                elif value == "":
                    formatted_value = "<em>N/A</em>"
                else:
                    formatted_value = str(value)
                
                html += f"<td>{formatted_value}</td>\n"
            html += "</tr>\n"
        
        html += "</tbody>\n</table>"
        return html
    
    def _generate_lineage_graph_html(self, lineage_data: dict) -> str:
        """
        Generate an HTML representation of a lineage graph using embedded D3.js.
        
        Args:
            lineage_data: Dictionary containing nodes and edges
            
        Returns:
            HTML string with embedded D3.js visualization
        """
        nodes = lineage_data.get("nodes", [])
        edges = lineage_data.get("edges", [])
        
        if not nodes or not edges:
            return "<em>Insufficient data for lineage visualization</em>"
        
        # Create a unique ID for this graph (to avoid conflicts with multiple graphs)
        graph_id = f"lineage-graph-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the HTML with embedded D3.js
        html = f"""
        <div class="lineage-graph" id="{graph_id}">
          <script src="https://d3js.org/d3.v7.min.js"></script>
          <script>
          (function() {{
              // Data for the graph
              const nodes = {json.dumps(nodes)};
              const links = {json.dumps(edges)};
              
              // Create a D3 force simulation
              const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(400, 300));
              
              // Create the SVG container
              const svg = d3.select("#{graph_id}")
                .append("svg")
                .attr("viewBox", "0 0 800 600")
                .attr("preserveAspectRatio", "xMidYMid meet")
                .attr("width", "100%")
                .attr("height", "500px");
              
              // Define arrow markers
              svg.append("defs").append("marker")
                .attr("id", "{graph_id}-arrow")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 20)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("class", "arrowhead")
                .attr("d", "M0,-5L10,0L0,5");
              
              // Create links
              const link = svg.append("g")
                .selectAll("path")
                .data(links)
                .enter().append("path")
                .attr("class", "link")
                .attr("marker-end", "url(#{graph_id}-arrow)");
              
              // Create nodes
              const node = svg.append("g")
                .selectAll("g")
                .data(nodes)
                .enter().append("g")
                .attr("class", "node")
                .call(d3.drag()
                  .on("start", dragstarted)
                  .on("drag", dragged)
                  .on("end", dragended));
              
              // Add circles to nodes
              node.append("circle")
                .attr("r", 10)
                .attr("fill", function(d) {{
                  if (d.type === "source") return "#34a853";
                  if (d.type === "target") return "#ea4335";
                  if (d.type === "transformation") return "#fbbc05";
                  return "#4285f4";
                }});
              
              // Add labels
              node.append("text")
                .attr("dx", 12)
                .attr("dy", ".35em")
                .text(d => d.name);
              
              // Add tooltips
              node.append("title")
                .text(function(d) {{
                  return "Name: " + d.name + "\\nTable: " + (d.table || "Unknown") + "\\nType: " + (d.type || "Data Element");
                }});
              
              // Update on tick
              simulation.on("tick", function() {{
                link.attr("d", function(d) {{
                  return "M" + d.source.x + "," + d.source.y + " L" + d.target.x + "," + d.target.y;
                }});
                
                node.attr("transform", function(d) {{
                  return "translate(" + d.x + "," + d.y + ")";
                }});
              }});
              
              // Drag functions
              function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
              }}
              
              function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
              }}
              
              function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
              }}
            }})();
          </script>
        </div>
        """
        
        return html
        
    def _format_response_for_display(self, response: str) -> str:
        """
        Format the agent response for better UI display.
        
        Args:
            response: Raw response from the agent
            
        Returns:
            Formatted response ready for UI display
        """
        # First, ensure no ReAct format markers are present (clean_react_output should have done this already)
        clean_response = response.strip()
        
        # Extract and format any embedded JSON data
        import re
        import json
        
        # Add CSS styles for tables and graphs
        css_styles = self._get_css_styles()
        
        # Check for lineage data that should be formatted as a graph visualization
        if "nodes" in clean_response and "edges" in clean_response and "lineage" in clean_response.lower():
            try:
                # Look for lineage data in JSON format
                lineage_pattern = r'```(?:json)?\s*(\{[\s\S]*?"nodes"[\s\S]*?"edges"[\s\S]*?\})```'
                lineage_match = re.search(lineage_pattern, clean_response)
                if lineage_match:
                    lineage_data = json.loads(lineage_match.group(1))
                    if "nodes" in lineage_data and "edges" in lineage_data:
                        graph_html = self._generate_lineage_graph_html(lineage_data)
                        # Replace the JSON with a message about the graph and the graph HTML
                        graph_message = "\n\n### Lineage Graph Visualization\n\n"
                        graph_message += "The lineage graph has been generated with the following elements:\n"
                        graph_message += f"- Nodes: {len(lineage_data.get('nodes', []))} data elements\n"
                        graph_message += f"- Connections: {len(lineage_data.get('edges', []))} data flows\n\n"
                        graph_message += graph_html
                        clean_response = clean_response.replace(lineage_match.group(0), graph_message)
            except Exception as e:
                logger.warning(f"Failed to generate lineage graph: {e}")
        
        # Look for JSON patterns in the text
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])```'
        json_matches = re.finditer(json_pattern, clean_response)
        
        # Process each JSON block found
        for match in json_matches:
            try:
                json_text = match.group(1)
                parsed_json = json.loads(json_text)
                
                # Create a nicely formatted representation
                if isinstance(parsed_json, dict):
                    formatted_output = self._format_json_as_table(parsed_json)
                elif isinstance(parsed_json, list) and len(parsed_json) > 0:
                    formatted_output = self._format_list_as_table(parsed_json)
                else:
                    formatted_output = self._format_json_dict(parsed_json) if isinstance(parsed_json, dict) else self._format_json_list(parsed_json)
                
                # Replace the JSON block with the formatted output
                clean_response = clean_response.replace(match.group(0), formatted_output)
            except json.JSONDecodeError:
                # If it's not valid JSON, leave it as is
                continue
        
        # Remove any remaining markdown code blocks since we've processed them
        clean_response = re.sub(r'```(?:json|python)?([\s\S]*?)```', r'\1', clean_response)
        
        # Add paragraph spacing for better readability
        paragraphs = [p for p in clean_response.split("\n\n") if p.strip()]
        formatted_text = "\n\n".join(paragraphs)
        
        # Add CSS styles at the beginning for HTML display
        if "<table>" in formatted_text or '<div class="lineage-graph">' in formatted_text:
            formatted_text = css_styles + formatted_text
            
        return formatted_text
        
    def _format_json_dict(self, data: dict, indent=0) -> str:
        """Format a dictionary for display."""
        if not data:
            return "{}"
            
        result = []
        indent_str = " " * indent
        
        # For nested data, we want a cleaner presentation
        for key, value in data.items():
            if isinstance(value, dict):
                if len(value) > 3:  # For complex dicts, summarize
                    result.append(f"{indent_str}• {key}: {{...}} ({len(value)} items)")
                else:
                    result.append(f"{indent_str}• {key}:")
                    result.append(self._format_json_dict(value, indent + 2))
            elif isinstance(value, list):
                if len(value) > 3:  # For long lists, summarize
                    result.append(f"{indent_str}• {key}: [...] ({len(value)} items)")
                else:
                    result.append(f"{indent_str}• {key}:")
                    result.append(self._format_json_list(value, indent + 2))
            else:
                result.append(f"{indent_str}• {key}: {value}")
                
        return "\n".join(result)
        
    def _format_json_list(self, data: list, indent=0) -> str:
        """Format a list for display."""
        if not data:
            return "[]"
            
        result = []
        indent_str = " " * indent
        
        # Convert each list item based on its type
        for item in data:
            if isinstance(item, dict):
                if len(item) > 3:  # For complex items, summarize
                    result.append(f"{indent_str}- {{...}} ({len(item)} properties)")
                else:
                    result.append(f"{indent_str}-")
                    result.append(self._format_json_dict(item, indent + 2))
            elif isinstance(item, list):
                if len(item) > 3:  # For nested lists, summarize
                    result.append(f"{indent_str}- [...] ({len(item)} items)")
                else:
                    result.append(f"{indent_str}-")
                    result.append(self._format_json_list(item, indent + 2))
            else:
                result.append(f"{indent_str}- {item}")
                
        return "\n".join(result)
        
    def _format_json_as_table(self, data: dict) -> str:
        """Format a dictionary as a structured table."""
        if not data:
            return "{}"
        
        # If this looks like a result with success/error pattern
        if "success" in data:
            if not data.get("success", True):
                return f"Error: {data.get('error', 'Unknown error')}"
            
            # Remove success flag for display
            display_data = {k: v for k, v in data.items() if k != "success"}
        else:
            display_data = data
        
        # For simple key-value pairs
        if not any(isinstance(v, (dict, list)) for v in display_data.values()):
            # Create a simple two-column table
            result = "\n| Property | Value |\n| --- | --- |\n"
            for key, value in display_data.items():
                result += f"| {key} | {value} |\n"
            return result
        
        # For complex nested structures, format each section separately
        result = []
        for key, value in display_data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # Format list of dictionaries as a table
                result.append(f"\n### {key.title()}")
                result.append(self._format_list_as_table(value))
            elif isinstance(value, dict):
                # Format nested dictionary
                result.append(f"\n### {key.title()}")
                result.append(self._format_json_as_table(value))
            elif isinstance(value, list):
                # Format simple list
                result.append(f"\n### {key.title()}")
                result.append(", ".join(str(item) for item in value))
            else:
                # Format simple value
                result.append(f"\n### {key.title()}")
                result.append(str(value))
                
        return "\n".join(result)
    
    def _format_list_as_table(self, data: list) -> str:
        """Format a list of dictionaries as a table."""
        if not data or not isinstance(data[0], dict):
            return self._format_json_list(data)
        
        # Get all unique keys across all dictionaries
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        # If too many columns, select the most important ones
        if len(all_keys) > 6:
            # Priority keys (customize based on your domain)
            priority_keys = ['id', 'name', 'code', 'description', 'query_code', 'element_code', 'source', 'target', 'table', 'transformation']
            selected_keys = [k for k in priority_keys if k in all_keys]
            
            # Add a few more if we still have space
            remaining_keys = list(all_keys - set(selected_keys))
            selected_keys.extend(remaining_keys[:6 - len(selected_keys)])
        else:
            selected_keys = sorted(all_keys)
            
        # Create table header
        header = "| " + " | ".join(k.replace('_', ' ').title() for k in selected_keys) + " |"
        separator = "| " + " | ".join(['---'] * len(selected_keys)) + " |"
        
        # Create table rows
        rows = []
        for item in data:
            if isinstance(item, dict):
                row = "| " + " | ".join(str(item.get(key, "")) for key in selected_keys) + " |"
                rows.append(row)
            else:
                # Handle non-dict items
                rows.append(f"| {item} |")
        
        return "\n".join([header, separator] + rows)
    
    def _generate_lineage_graph_html(self, lineage_data: dict) -> str:
        """
        Generate an HTML representation of the lineage graph.
        
        Args:
            lineage_data: Dictionary containing nodes and edges
            
        Returns:
            HTML representation of the graph
        """
        try:
            nodes = lineage_data.get('nodes', [])
            edges = lineage_data.get('edges', [])
            
            if not nodes:
                return "No nodes found in lineage data."
                
            # Generate a simple ASCII representation for text-based UI
            # In a real application, you'd return JavaScript for visualization
            result = ["```", "LINEAGE GRAPH VISUALIZATION", ""]
            
            # List all nodes
            result.append("Nodes:")
            for i, node in enumerate(nodes):
                node_id = node.get('id', f'unknown_{i}')
                name = node.get('name', 'Unknown')
                table = node.get('table', 'Unknown')
                result.append(f"  {node_id}: {name} ({table})")
            
            result.append("")
            
            # List all edges
            result.append("Connections:")
            for edge in edges:
                source = edge.get('source', 'unknown')
                target = edge.get('target', 'unknown')
                query = edge.get('query', 'Unknown')
                transformation = edge.get('transformation', 'Unknown')
                
                # Find source and target node names
                source_name = next((n.get('name', 'Unknown') for n in nodes if n.get('id') == source), source)
                target_name = next((n.get('name', 'Unknown') for n in nodes if n.get('id') == target), target)
                
                result.append(f"  {source_name} → {target_name} via {query}")
                if transformation and transformation != 'Unknown':
                    result.append(f"    Transformation: {transformation}")
            
            result.append("```")
            
            # For web UI display, you'd return a dynamic visualization:
            html = """
            <div class="lineage-graph">
                <div class="graph-legend">
                    <div class="legend-item"><span class="node-dot"></span> Data Elements</div>
                    <div class="legend-item"><span class="edge-line"></span> Data Flows</div>
                </div>
                <div class="graph-container">
                    <pre>{text_graph}</pre>
                </div>
                <div class="graph-instructions">
                    <p>For interactive visualization, export this lineage data to a graph visualization tool.</p>
                </div>
            </div>
            """.format(text_graph="\n".join(result))
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating lineage graph: {e}")
            return f"Error generating lineage graph visualization: {str(e)}"
        
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract key entities from user message for memory storage.
        
        Args:
            message: User's input message
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Simple regex-based entity extraction
        # Extract contract names (assume they contain "contract" or "pipeline")
        import re
        contract_pattern = r'(?:contract|pipeline)\s+(?:named|called|for)?\s*[\'"]?([A-Za-z0-9\s]+)[\'"]?'
        contract_matches = re.findall(contract_pattern, message, re.IGNORECASE)
        if contract_matches:
            entities["contract"] = contract_matches[0].strip()
        
        # Extract element names (data fields)
        element_pattern = r'(?:field|column|element|attribute)\s+(?:named|called)?\s*[\'"]?([A-Za-z0-9_]+)[\'"]?'
        element_matches = re.findall(element_pattern, message, re.IGNORECASE)
        if element_matches:
            entities["element"] = element_matches[0].strip()
            
        # Extract query/ETL codes (like Q001)
        query_pattern = r'(?:query|etl|pipeline)\s+(?:code|id)?\s*[\'"]?([Q][0-9]{3})[\'"]?'
        query_matches = re.findall(query_pattern, message, re.IGNORECASE)
        if query_matches:
            entities["query_code"] = query_matches[0].strip()
            
        # Extract direction for lineage tracing
        if any(term in message.lower() for term in ["upstream", "source", "input", "comes from"]):
            entities["direction"] = "upstream"
        elif any(term in message.lower() for term in ["downstream", "target", "output", "flows to"]):
            entities["direction"] = "downstream"
        elif any(term in message.lower() for term in ["bidirectional", "both directions", "full path"]):
            entities["direction"] = "bidirectional"
            
        return entities
        
    def _update_memory(self, entities: Dict[str, Any]):
        """
        Update the agent's memory with extracted entities.
        
        Args:
            entities: Dictionary of entities to store in memory
        """
        # Add entities to conversation state memory
        if "memory" not in self.conversation_state:
            self.conversation_state["memory"] = {}
            
        for key, value in entities.items():
            self.conversation_state["memory"][key] = value
            
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.7) -> bool:
        """
        Perform fuzzy matching between query and target strings.
        
        Args:
            query: User query string
            target: Target metadata string to match against
            threshold: Similarity threshold (0.0-1.0) where 1.0 is exact match
            
        Returns:
            Boolean indicating whether the strings match according to the threshold
        """
        # Simple word overlap score
        query_words = set(query.lower().split())
        target_words = set(target.lower().split())
        
        # Skip very short targets
        if len(target_words) < 2:
            return False
            
        # Calculate word overlap
        common_words = query_words.intersection(target_words)
        if not common_words:
            return False
            
        # Calculate similarity score
        similarity = len(common_words) / max(len(query_words), len(target_words))
        
        # Look for consecutive word matches which are stronger indicators
        query_bigrams = self._get_bigrams(query.lower())
        target_bigrams = self._get_bigrams(target.lower())
        common_bigrams = set(query_bigrams).intersection(set(target_bigrams))
        
        # Boost score if we have consecutive word matches
        if common_bigrams:
            similarity += 0.2 * (len(common_bigrams) / max(len(query_bigrams), len(target_bigrams)))
            
        return similarity >= threshold
        
    def _get_bigrams(self, text: str) -> List[str]:
        """Get bigrams (consecutive word pairs) from text"""
        words = text.split()
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        
    def _search_lineage_db(self, query: str) -> Dict[str, Any]:
        """
        Search for relevant metadata in the lineage database
        
        Args:
            query: User query string
            
        Returns:
            Dict with search results
        """
        # Use the search_lineage_database tool
        try:
            # Call the tool with the correct parameter name
            return search_lineage_database(query)
        except Exception as e:
            logger.error(f"Error searching lineage database: {e}")
            return {"success": False, "error": str(e)}
            
    def _get_memory_context(self, current_query: str) -> Dict[str, Any]:
        """
        Build memory context to enhance the current query.
        
        Args:
            current_query: The current user query
            
        Returns:
            Dictionary with memory context
        """
        context = {}
        
        # Get conversation history from memory
        memory_vars = self.memory.load_memory_variables({})
        
        # Add properly formatted conversation history
        if memory_vars:
            if "history" in memory_vars:
                context["conversation_history"] = memory_vars
            elif "messages" in memory_vars:
                # Format the messages into history format for compatibility
                context["conversation_history"] = {"history": []}
                messages = memory_vars["messages"]
                for i in range(0, len(messages)-1, 2):
                    if i+1 < len(messages):
                        user_msg = messages[i].content if hasattr(messages[i], "content") else str(messages[i])
                        ai_msg = messages[i+1].content if hasattr(messages[i+1], "content") else str(messages[i+1])
                        context["conversation_history"]["history"].append({"input": user_msg, "output": ai_msg})
            else:
                # Try to normalize whatever memory format we have
                logger.warning(f"Unexpected memory format: {memory_vars}")
                context["conversation_history"] = {"history": memory_vars}
        
        # Add stored entities if they exist
        if "memory" in self.conversation_state:
            context["entities"] = self.conversation_state["memory"]
            
        # Try to determine if the current query references previous context
        if any(term in current_query.lower() for term in [
            "it", "that", "this", "these", "those", "they", "them", "its", "their",
            "previous", "before", "earlier"
        ]):
            context["references_previous"] = True
            
        return context
    
    def _verify_metadata_match(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the user's query is asking about metadata that exists in our database.
        Uses both exact matching and fuzzy matching to find relevant metadata.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (match_found, available_options) where:
            - match_found: Boolean indicating if query matches existing metadata
            - available_options: Dict with potential matching entities as options
        """
        logger.info("Verifying if query matches available metadata")
        
        # First try direct tool-based search
        search_results = self._search_lineage_db(query)
        if search_results.get("success", False) and search_results.get("total_results", 0) > 0:
            logger.info(f"Found {search_results.get('total_results')} results using search_lineage_database tool")
            
            # Extract results into our format
            results = search_results.get("results", {})
            
            potential_contracts = [
                {
                    "code": contract.get("code"),
                    "name": contract.get("name"),
                    "description": contract.get("description"),
                    "match_type": "search"
                } for contract in results.get("contracts", [])
            ]
            
            potential_elements = [
                {
                    "code": element.get("code"),
                    "name": element.get("name"),
                    "table": element.get("table"),
                    "definition": element.get("definition", ""),
                    "match_type": "search"
                } for element in results.get("elements", [])
            ]
            
            potential_queries = [
                {
                    "code": query_result.get("code"),
                    "description": query_result.get("description"),
                    "source": "",
                    "target": "",
                    "match_type": "search"
                } for query_result in results.get("pipelines", [])
            ]
            
            potential_transformations = [
                {
                    "code": transform.get("code"),
                    "rules": transform.get("rules"),
                    "match_type": "search"
                } for transform in results.get("transformations", [])
            ]
            
            # If we have significant results, return them
            if (len(potential_contracts) + len(potential_elements) + 
                len(potential_queries) + len(potential_transformations) > 0):
                return True, {
                    "contracts": potential_contracts,
                    "elements": potential_elements,
                    "queries": potential_queries,
                    "transformations": potential_transformations
                }
        
        # Fall back to direct database querying for more comprehensive results
        conn = db_manager_global.get_connection()
        cursor = conn.cursor()
        
        # Lists to store potential matches
        potential_contracts = []
        potential_elements = []
        potential_queries = []
        potential_transformations = []
        
        # Check if query is asking about specific metadata types
        metadata_type_keywords = {
            "contract": ["contract", "contracts", "agreement", "pipeline", "data flow"],
            "element": ["element", "column", "field", "attribute", "data field", "table column"],
            "query": ["etl", "query", "pipeline", "process", "sql", "transformation", "job", "load"],
            "transformation": ["transformation", "transform", "convert", "rule", "mapping"]
        }
        
        # Determine what type of metadata the user is likely asking about
        metadata_types = []
        for metadata_type, keywords in metadata_type_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                metadata_types.append(metadata_type)
                
        # If we can identify the type, we should focus our search on that type
        focused_search = len(metadata_types) > 0
        logger.info(f"Detected metadata types: {metadata_types}")
        
        # Get all contracts
        cursor.execute("SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts")
        all_contracts = cursor.fetchall()
        contract_names = [c[1].lower() if c[1] else "" for c in all_contracts]
        
        # Get all elements
        cursor.execute("""
            SELECT bem.v_data_element_code, bem.v_data_element_name, bem.v_table_name, 
                   bd.v_business_definition 
            FROM business_element_mapping bem
            LEFT JOIN business_dictionary bd ON bem.v_business_element_code = bd.v_business_element_code
        """)
        all_elements = cursor.fetchall()
        element_names = [e[1].lower() if e[1] else "" for e in all_elements]
        
        # Get all queries
        cursor.execute("""
            SELECT v_query_code, v_query_description, v_target_table_or_object, v_source_table_or_object 
            FROM etl_pipeline_metadata
        """)
        all_queries = cursor.fetchall()
        query_codes = [q[0].lower() for q in all_queries]
        query_descs = [q[1].lower() if q[1] else "" for q in all_queries]
        
        # Get all transformation rules
        cursor.execute("SELECT v_transformation_code, v_transformation_rules FROM transformation_rules")
        all_transformations = cursor.fetchall()
        transformation_codes = [t[0].lower() for t in all_transformations]
        transformation_rules = [t[1].lower() if t[1] else "" for t in all_transformations]
        
        # Check for direct mentions of entities in the query
        query_lower = query.lower()
        # Connect to DB and get all available entities for matching
        conn = db_manager_global.get_connection()
        cursor = conn.cursor()
        
        # Get all contract names
        cursor.execute("SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts")
        all_contracts = cursor.fetchall()
        contract_names = [c[1].lower() for c in all_contracts]
        
        # Get all element names
        cursor.execute("SELECT v_data_element_code, v_data_element_name, v_table_name FROM business_element_mapping")
        all_elements = cursor.fetchall()
        element_names = [e[1].lower() for e in all_elements]
        
        # Get all query codes and descriptions
        cursor.execute("SELECT v_query_code, v_query_description FROM etl_pipeline_metadata")
        all_queries = cursor.fetchall()
        query_codes = [q[0].lower() for q in all_queries]
        query_descs = [q[1].lower() if q[1] else "" for q in all_queries]
        
        # Get all transformation rules
        cursor.execute("SELECT v_transformation_code, v_transformation_rules FROM transformation_rules")
        all_transformations = cursor.fetchall()
        transformation_codes = [t[0].lower() for t in all_transformations]
        transformation_rules = [t[1].lower() if t[1] else "" for t in all_transformations]
        
        # Check for direct mentions of entities in the query
        query_lower = query.lower()
        
        # Check for contract mentions
        for i, contract_name in enumerate(contract_names):
            if contract_name and contract_name in query_lower:
                potential_contracts.append({
                    "code": all_contracts[i][0],
                    "name": all_contracts[i][1],
                    "description": all_contracts[i][2]
                })
        
        # Check for element mentions
        for i, element_name in enumerate(element_names):
            if element_name and element_name in query_lower:
                potential_elements.append({
                    "code": all_elements[i][0],
                    "name": all_elements[i][1],
                    "table": all_elements[i][2]
                })
                
        # Check for query code mentions
        for i, code in enumerate(query_codes):
            if code and code in query_lower:
                potential_queries.append({
                    "code": all_queries[i][0],
                    "description": all_queries[i][1]
                })
                
        # Check for query description mentions
        for i, desc in enumerate(query_descs):
            if desc and desc in query_lower:
                # Only add if not already added by code
                query_code = all_queries[i][0]
                if not any(q["code"] == query_code for q in potential_queries):
                    potential_queries.append({
                        "code": query_code,
                        "description": all_queries[i][1]
                    })
                    
        # Check for transformation mentions
        for i, code in enumerate(transformation_codes):
            if code and code in query_lower:
                potential_transformations.append({
                    "code": all_transformations[i][0],
                    "rules": all_transformations[i][1]
                })
                
        # Check for transformation rule mentions
        for i, rule in enumerate(transformation_rules):
            if rule and rule in query_lower:
                # Only add if not already added by code
                transformation_code = all_transformations[i][0]
                if not any(t["code"] == transformation_code for t in potential_transformations):
                    potential_transformations.append({
                        "code": transformation_code,
                        "rules": all_transformations[i][1]
                    })
        
        # If no exact matches found or specific metadata types identified, try fuzzy matching
        if (len(potential_contracts) + len(potential_elements) + 
            len(potential_queries) + len(potential_transformations) < 3) or focused_search:
            
            # If we identified specific metadata types, focus on those
            if not metadata_types or "contract" in metadata_types:
                # Fuzzy match contracts
                for i, contract in enumerate(all_contracts):
                    code, name, desc = contract
                    if (name and self._fuzzy_match(query_lower, name.lower())) or \
                       (desc and self._fuzzy_match(query_lower, desc.lower())):
                        if not any(c["code"] == code for c in potential_contracts):
                            potential_contracts.append({
                                "code": code,
                                "name": name,
                                "description": desc,
                                "match_type": "fuzzy"
                            })
            
            if not metadata_types or "element" in metadata_types:
                # Fuzzy match elements
                for i, element in enumerate(all_elements):
                    code, name, table = element[0], element[1], element[2]
                    definition = element[3] if len(element) > 3 and element[3] else "No definition available"
                    
                    if (name and self._fuzzy_match(query_lower, name.lower())) or \
                       (table and self._fuzzy_match(query_lower, table.lower())) or \
                       (definition and self._fuzzy_match(query_lower, definition.lower())):
                        if not any(e["code"] == code for e in potential_elements):
                            potential_elements.append({
                                "code": code,
                                "name": name,
                                "table": table,
                                "definition": definition,
                                "match_type": "fuzzy"
                            })
            
            if not metadata_types or "query" in metadata_types:
                # Fuzzy match queries
                for i, query_item in enumerate(all_queries):
                    code, desc = query_item[0], query_item[1]
                    target = query_item[2] if len(query_item) > 2 else ""
                    source = query_item[3] if len(query_item) > 3 else ""
                    
                    if (desc and self._fuzzy_match(query_lower, desc.lower())) or \
                       (target and self._fuzzy_match(query_lower, target.lower())) or \
                       (source and self._fuzzy_match(query_lower, source.lower())):
                        if not any(q["code"] == code for q in potential_queries):
                            potential_queries.append({
                                "code": code,
                                "description": desc,
                                "target": target,
                                "source": source,
                                "match_type": "fuzzy"
                            })
            
            if not metadata_types or "transformation" in metadata_types:
                # Fuzzy match transformations
                for i, transformation in enumerate(all_transformations):
                    code, rules = transformation
                    if rules and self._fuzzy_match(query_lower, rules.lower()):
                        if not any(t["code"] == code for t in potential_transformations):
                            potential_transformations.append({
                                "code": code,
                                "rules": rules,
                                "match_type": "fuzzy"
                            })
        
        # Close DB connection
        conn.close()
        
        # Determine if we found any matches
        available_options = {
            "contracts": potential_contracts,
            "elements": potential_elements,
            "queries": potential_queries,
            "transformations": potential_transformations
        }
        
        # Consider a match found if we have any potential entities
        match_found = (len(potential_contracts) > 0 or 
                      len(potential_elements) > 0 or 
                      len(potential_queries) > 0 or
                      len(potential_transformations) > 0)
        
        # Debug logging
        logger.info(f"Metadata match found: {match_found}")
        logger.info(f"Available options: contracts={len(potential_contracts)}, elements={len(potential_elements)}, " +
                  f"queries={len(potential_queries)}, transformations={len(potential_transformations)}")
        
        return match_found, available_options
        
    def _get_metadata_options(self, query: str, available_options: Dict[str, Any]) -> str:
        """
        Format available metadata options for user selection when query doesn't exactly match.
        
        Args:
            query: Original user query
            available_options: Dict with potential matching entities
            
        Returns:
            Formatted string with options for the user to select from
        """
        contracts = available_options.get("contracts", [])
        elements = available_options.get("elements", [])
        queries = available_options.get("queries", [])
        transformations = available_options.get("transformations", [])
        
        options_text = []
        options_text.append("I need more specificity to answer your query accurately. Please select from these metadata options to help me provide the most relevant information:")
        options_text.append("\nLEGEND: ✓ = Exact match, ≈ = Similar match")
        
        if contracts:
            options_text.append("\n**Available Contracts:**")
            for i, contract in enumerate(contracts[:5]):  # Limit to 5 options
                match_indicator = "✓" if contract.get('match_type') == "exact" else "≈"
                options_text.append(f"{i+1}. {match_indicator} {contract['name']} ({contract['code']}) - {contract['description']}")
            if len(contracts) > 5:
                options_text.append(f"...and {len(contracts) - 5} more contracts")
                
        if elements:
            options_text.append("\n**Available Data Elements:**")
            for i, element in enumerate(elements[:5]):  # Limit to 5 options
                match_indicator = "✓" if element.get('match_type') == "exact" else "≈"
                definition = f" - {element.get('definition', '')}" if element.get('definition') else ""
                options_text.append(f"{i+1}. {match_indicator} {element['name']} (Table: {element['table']}, Code: {element['code']}){definition}")
            if len(elements) > 5:
                options_text.append(f"...and {len(elements) - 5} more elements")
                
        if queries:
            options_text.append("\n**Available ETL Pipelines:**")
            for i, q in enumerate(queries[:5]):  # Limit to 5 options
                match_indicator = "✓" if q.get('match_type') == "exact" else "≈"
                source_target = f" ({q.get('source', '')} → {q.get('target', '')})" if q.get('source') and q.get('target') else ""
                options_text.append(f"{i+1}. {match_indicator} {q['code']} - {q['description']}{source_target}")
            if len(queries) > 5:
                options_text.append(f"...and {len(queries) - 5} more pipelines")
                
        if transformations:
            options_text.append("\n**Available Transformation Rules:**")
            for i, t in enumerate(transformations[:5]):  # Limit to 5 options
                match_indicator = "✓" if t.get('match_type') == "exact" else "≈"
                # Truncate rules if too long
                rules = t['rules']
                if len(rules) > 50:
                    rules = rules[:50] + "..."
                options_text.append(f"{i+1}. {match_indicator} {t['code']} - {rules}")
            if len(transformations) > 5:
                options_text.append(f"...and {len(transformations) - 5} more transformations")
                
        if not contracts and not elements and not queries and not transformations:
            options_text.append("\nNo related metadata found. Please try a different query or provide more specific information.")
            options_text.append("You can ask about available contracts, data elements, ETL pipelines, or transformations to see what's available.")
        
        options_text.append("\nPlease select an option or refine your query to be more specific.")
        
        return "\n".join(options_text)
    
    def _detect_uncertainty(self, response: str) -> bool:
        """
        Detect if the response indicates uncertainty or inability to determine an answer.
        
        Args:
            response: Agent's response string
            
        Returns:
            True if uncertainty is detected, False otherwise
        """
        # Common uncertainty phrases
        uncertainty_phrases = [
            "I'm not sure",
            "I cannot determine",
            "I don't have enough information",
            "It's unclear",
            "I'm unable to",
            "I don't know",
            "I'm uncertain",
            "I'm having trouble",
            "cannot be determined",
            "more information is needed",
            "I need additional information",
            "unclear from the available data",
            "ambiguous",
            "insufficient data",
            "difficult to say",
            "hard to determine",
            "there's no way to know",
            "can't be certain",
            "would need more context",
            "missing details",
            "incomplete information",
            "not possible to determine",
            "without more details",
            "not clear from the information",
            "would need to know",
            "need clarification",
            "to answer properly I would need",
            "not enough context",
            "difficult to answer without",
            "can you provide more details",
            "to give a complete answer",
            "to accurately answer",
            "I would need more information",
            "can't provide a definitive answer"
        ]
        
        # Phrases that indicate the agent is making assumptions
        assumption_phrases = [
            "I assume",
            "assuming that",
            "based on assumptions",
            "I'm guessing",
            "my best guess",
            "it seems like",
            "possibly",
            "might be",
            "could be",
            "perhaps",
            "probably",
            "likely",
            "appears to be",
            "it may be that",
            "presumably",
            "it's possible that",
            "it looks like",
            "seems to be",
            "potentially",
            "apparently"
        ]
        
        # Check for uncertainty markers
        response_lower = response.lower()
        
        # Explicit uncertainty markers
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            logger.info("Explicit uncertainty detected in agent response")
            return True
            
        # Check for questions in the response (agent asking for clarification)
        if "?" in response and any(phrase in response_lower for phrase in [
            "do you mean",
            "could you clarify",
            "would you like",
            "can you provide",
            "can you specify",
            "which one",
            "did you want",
            "are you referring to",
            "did you mean",
            "could you provide",
            "would you be able to",
            "can you clarify",
            "can you tell me more",
            "what specific",
            "which specific",
            "would you prefer"
        ]):
            logger.info("Agent asking clarification questions in response")
            return True
            
        # Check for multiple options presented (agent not sure which to pick)
        if any(word in response_lower for word in ["option", "alternative", "possibilities", "scenarios", "interpretations"]) and any(marker in response for marker in [
            "1.", "2.", "Option 1", "Option 2", "First,", "Second,", "A)", "B)", "•", "-", "I can think of", "There are several"
        ]):
            logger.info("Agent presenting multiple options in response")
            return True
            
        # Check for assumptions being made (if > 1 assumption phrases)
        assumption_count = sum(1 for phrase in assumption_phrases if phrase in response_lower)
        if assumption_count >= 1:
            logger.info(f"Agent making assumptions ({assumption_count}) in response")
            return True
            
        # Check for sentences starting with speculative words
        speculative_starters = ["if ", "assuming ", "in case ", "should ", "when "]
        sentence_count = sum(1 for sentence in response.split(". ") 
                          if any(sentence.lower().strip().startswith(starter) for starter in speculative_starters))
        if sentence_count >= 2:
            logger.info(f"Agent using multiple speculative sentences ({sentence_count}) in response")
            return True
            
        return False
    
    def _get_user_feedback(self, message: str, uncertain_response: str) -> str:
        """
        Get feedback from the human user when the agent is uncertain.
        In CLI mode, this directly prompts for input.
        In API/web mode, this could signal the UI to prompt the user for additional input.
        
        Args:
            message: Original user message
            uncertain_response: The uncertain response from the agent
            
        Returns:
            User feedback string
        """
        # Check if human feedback is enabled
        if not self.enable_human_feedback:
            logger.info("Human feedback is disabled, continuing without feedback")
            return ""
            
        if self.interface_mode == "cli":
            # In CLI mode, directly ask for input
            print("\n" + "=" * 60)
            print("I need your help to better answer your question:")
            print("=" * 60)
            print(f"Your question: {message}")
            print("\nMy initial understanding:")
            print(uncertain_response)
            print("\nCould you please provide additional information or clarification?")
            feedback = input("Your feedback: ").strip()
            return feedback
        elif self.interface_mode == "web":
            # In web mode, we would typically raise an event or return a special response
            # that the web UI would interpret as a request for clarification
            # This is a placeholder for web implementation - would need integration with the actual UI
            logger.info("Web interface feedback requested - this would trigger a UI prompt")
            
            # For now, we'll return a placeholder. In a real implementation,
            # we might raise a custom exception or return a special object that
            # the web handler would detect and handle appropriately
            return "WEB_FEEDBACK_REQUIRED"
        else:
            # Default to no feedback for unknown interfaces
            logger.warning(f"Unknown interface mode: {self.interface_mode}, continuing without feedback")
            return ""
        
    def _refine_response_with_feedback(self, original_message: str, uncertain_response: str, user_feedback: str) -> str:
        """
        Refine the agent's response using human feedback.
        
        Args:
            original_message: Original user message
            uncertain_response: Initial uncertain response from the agent
            user_feedback: User's feedback/clarification
            
        Returns:
            Refined agent response
        """
        # Check if the feedback might be selecting from metadata options
        metadata_selection = False
        selected_entity = None
        
        # Look for patterns that suggest the user is selecting a metadata option
        if re.search(r"^[1-5]\b", user_feedback.strip()):
            metadata_selection = True
            selected_option = user_feedback.strip()[0]
            logger.info(f"User appears to be selecting option {selected_option} from metadata choices")
            
        elif any(pattern in user_feedback.lower() for pattern in ["option", "select", "choose", "number", "#"]):
            metadata_selection = True
            logger.info("User appears to be making a selection from metadata choices")
            
        elif any(re.search(rf"\b{code}\b", user_feedback, re.IGNORECASE) for code in ["Q001", "Q002", "Q003", "DE001", "C001"]):
            metadata_selection = True
            # Extract potential code from feedback
            codes = re.findall(r'\b([CQD]E?\d{3})\b', user_feedback, re.IGNORECASE)
            if codes:
                selected_entity = codes[0].upper()
                logger.info(f"User appears to be selecting entity code: {selected_entity}")
        
        # Create an enhanced prompt based on the type of feedback
        if metadata_selection:
            # Create a prompt focused on the selected metadata entity
            refinement_prompt = f"""
I initially responded to the user's question:
"{original_message}"

I indicated that I couldn't find an exact match in the metadata, and offered some options.

The user has now selected a specific metadata entity:
"{user_feedback}"

GUIDELINES FOR YOUR REFINED ANSWER:
1. Focus EXCLUSIVELY on the metadata entity the user has selected
2. First, use the appropriate tools to query for information about this specific entity
3. ONLY provide information that is explicitly available in the metadata
4. DO NOT hallucinate or infer information that isn't directly supported by the metadata
5. If certain details aren't available in the metadata, clearly state this limitation
6. Make your response conversational and helpful
7. Support all statements with specific data from the lineage database

Remember: It is critically important to avoid making up information and to only present facts that are directly supported by the metadata.
"""
        else:
            # Standard refinement prompt for other types of feedback
            refinement_prompt = f"""
I initially responded to the user's question:
"{original_message}"

With this response indicating uncertainty:
"{uncertain_response}"

The user provided this additional information/clarification:
"{user_feedback}"

Using this additional context, please provide a more accurate and complete response.

GUIDELINES FOR YOUR REFINED ANSWER:
1. Acknowledge the additional information the user provided
2. Clearly explain how this new information helps address the original question
3. ONLY provide information that is explicitly available in the metadata database
4. DO NOT hallucinate or make up information that isn't found in the metadata
5. If some aspects still remain uncertain, clearly state what you can answer with confidence and what still requires clarification
6. Make your response conversational and helpful
7. Support your statements with specific data from the lineage database when available

Make sure to explain your reasoning and how the user's feedback helped clarify the answer.
"""
        
        # Get the existing conversation state without the uncertain response
        messages_without_uncertain_response = [
            msg for msg in self.conversation_state["messages"] 
            if not (isinstance(msg, AIMessage) and msg.content == uncertain_response)
        ]
        
        # Create a new state with the refinement prompt
        refinement_state = {
            "messages": messages_without_uncertain_response + [HumanMessage(content=refinement_prompt)],
            "memory": self.conversation_state.get("memory", {})
        }
        
        logger.info("Refining response with user feedback")
        result = self.agent(refinement_state, refinement_prompt)
        
        # Extract refined response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            refined_response = ai_messages[-1].content
            # Update the conversation state
            self.conversation_state = result
            return refined_response
            
        return "I'm still having trouble providing a good response. Please try rephrasing your question."
    
    def process_feedback(self, original_message: str, feedback: str) -> str:
        """
        Process feedback from user for an uncertain response.
        
        Args:
            original_message: The original query that generated uncertainty
            feedback: User feedback/clarification
            
        Returns:
            Refined response from the agent
        """
        logger.info(f"Processing feedback for message: '{original_message}', feedback: '{feedback}'")
        
        try:
            # Check if this is a metadata selection (format: "I select: [selection]")
            is_metadata_selection = feedback.startswith("I select:")
            
            # If it's a metadata selection, process it as a new query combining original + selection
            if is_metadata_selection:
                selection = feedback.replace("I select:", "").strip()
                logger.info(f"Detected metadata selection: '{selection}' for query: '{original_message}'")
                
                # Create a new message that combines the original query with the selection
                combined_query = f"{original_message} [Selected: {selection}]"
                logger.info(f"Processing combined query: '{combined_query}'")
                
                # Process this combined query as a new message
                return self.process_message(combined_query)
                
            # For regular feedback, find the most recent uncertain response for this query
            recent_responses = [msg for msg in self.conversation_state["messages"] 
                               if isinstance(msg, AIMessage)]
            
            if not recent_responses:
                logger.warning("No previous AI messages found to refine")
                return "I couldn't find the previous conversation to refine. Could you please repeat your question?"
            
            uncertain_response = recent_responses[-1].content
            
            # Record the feedback in memory for future reference
            if "feedback_history" not in self.conversation_state.get("memory", {}):
                self.conversation_state.setdefault("memory", {})["feedback_history"] = []
            
            self.conversation_state["memory"]["feedback_history"].append({
                "timestamp": datetime.now().isoformat(),
                "query": original_message,
                "uncertain_response": uncertain_response,
                "feedback": feedback
            })
            
            # Refine the response with the feedback
            refined_response = self._refine_response_with_feedback(
                original_message, uncertain_response, feedback
            )
            
            # Save the refined response to memory
            self.memory.chat_memory.add_ai_message(refined_response)
            
            # Format the response for better UI presentation
            formatted_response = self._format_response_for_display(refined_response)
            
            logger.info(f"Refined response generated, length: {len(formatted_response)}")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            traceback.print_exc()
            return f"I encountered an error while processing your feedback: {str(e)}"
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: User message string
            
        Returns:
            Response from the agent
        """
        logger.info(f"Processing message: '{message}'")
        try:
            # First verify if the query matches available metadata
            metadata_match, available_options = self._verify_metadata_match(message)
            
            # If no metadata match found, offer options to the user
            if not metadata_match and self.enable_human_feedback:
                logger.info("No metadata match found for query, providing options to user")
                options_response = self._get_metadata_options(message, available_options)
                
                # For web interface, return a structured response
                if self.interface_mode == "web":
                    logger.info(f"Creating metadata selection response for web interface")
                    metadata_options = {
                        "needs_metadata_selection": True,
                        "original_query": message,
                        "options": available_options,
                        "formatted_options": options_response,
                        "reasoning": "The query does not match available metadata."
                    }
                    json_response = json.dumps(metadata_options)
                    logger.info(f"Metadata selection JSON response: {json_response[:100]}...")  # Log first 100 chars
                    return json_response
                
                # For CLI interface, return the formatted options
                return options_response
            
            # Extract entities from the message to store in memory
            entities = self._extract_entities(message)
            if entities:
                logger.info(f"Extracted entities: {entities}")
                # Update memory with extracted entities
                self._update_memory(entities)
            
            # Add message to memory
            self.memory.save_context({"input": message}, {"output": ""})
            
            # Retrieve memory to enhance context
            memory_context = self._get_memory_context(message)
            
            # Enhance memory context with metadata match information
            if "metadata_context" not in memory_context:
                memory_context["metadata_context"] = {}
                
            memory_context["metadata_context"]["has_matching_metadata"] = metadata_match
            if not metadata_match:
                memory_context["metadata_context"]["warning"] = "Query does not match available metadata. Be especially careful not to hallucinate information."
            
            # Add tool usage guidance to prevent hallucination
            tool_guidance = """
IMPORTANT: Always use available tools to query the metadata database for accurate information. 
DO NOT make assumptions or hallucinate information about:
- Contracts, pipelines, elements, or transformations that aren't explicitly found in the database
- Relationships between entities that aren't confirmed by tool results

Available search tools:
- search_lineage_database: For general searches across all metadata
- query_contract_by_name: When looking for specific contracts
- find_element_by_name: When looking for data elements/columns
- get_query_details: When looking for ETL pipeline details
"""
            
            # Use the simplified agent function directly with enhanced context
            initial_state = {
                "messages": self.conversation_state["messages"] + [
                    SystemMessage(content=tool_guidance),
                    HumanMessage(content=message)
                ],
                "memory": memory_context
            }
            
            logger.info("Calling agent function")
            result = self.agent(initial_state, message)
            logger.info("Agent function returned successfully")
            
            # Extract the final state
            self.conversation_state = result
            
            # Get the AI's response (last AI message)
            ai_messages = [msg for msg in self.conversation_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
                
                # Check for uncertainty or if no metadata match was found
                is_uncertain = self._detect_uncertainty(response) or not metadata_match
                
                # Check for uncertainty in the response and if human feedback is enabled
                if self.enable_human_feedback and is_uncertain:
                    logger.info("Detected uncertainty or no metadata match, requesting human feedback")
                    
                    # If the agent is uncertain, get human feedback
                    user_feedback = self._get_user_feedback(message, response)
                    
                    # Special handling for web mode
                    if self.interface_mode == "web" and user_feedback == "WEB_FEEDBACK_REQUIRED":
                        # In web mode, return a special response to trigger UI feedback
                        self.memory.save_context({"input": message}, {"output": response})
                        
                        # Add a marker that the UI can detect to request clarification
                        response = {
                            "needs_clarification": True,
                            "original_response": response,
                            "original_query": message,
                            "has_metadata_match": metadata_match,
                            "available_options": available_options if not metadata_match else {},
                            "reasoning": "The agent is uncertain about this response or the query doesn't match available metadata."
                        }
                        # For web mode, return the object as JSON for the web handler
                        return json.dumps(response)
                    
                    # For CLI mode or if we somehow got feedback from web
                    elif user_feedback:
                        logger.info("Received human feedback, refining response")
                        
                        # Record the feedback in memory for future reference
                        if "feedback_history" not in self.conversation_state.get("memory", {}):
                            self.conversation_state.setdefault("memory", {})["feedback_history"] = []
                        
                        self.conversation_state["memory"]["feedback_history"].append({
                            "timestamp": datetime.now().isoformat(),
                            "query": message,
                            "uncertain_response": response,
                            "feedback": user_feedback,
                            "has_metadata_match": metadata_match
                        })
                        
                        # Refine the response with the feedback
                        refined_response = self._refine_response_with_feedback(
                            message, response, user_feedback
                        )
                        
                        # Update response with refined version
                        response = refined_response
                        
                    # Save the refined response to memory
                    self.memory.save_context({"input": message}, {"output": response})
                
                # If no uncertainty or after refining, save the final response
                else:
                    # Add a warning if no metadata match was found but we're proceeding anyway
                    if not metadata_match:
                        response = ("⚠️ Note: Your query doesn't directly match available metadata. " +
                                   "The following response might be incomplete or less specific than desired.\n\n" +
                                   response)
                    
                    # Save the response to memory
                    self.memory.save_context({"input": message}, {"output": response})                # Format the response for better UI presentation
                formatted_response = self._format_response_for_display(response)
                
                logger.info(f"Agent response generated, length: {len(formatted_response)}")
                return formatted_response
            
            logger.warning("No AI message found in conversation state")
            return "I'm having trouble generating a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            traceback.print_exc()
            return f"I encountered an error while processing your request: {str(e)}"


def main():
    """Main function for an interactive CLI testing experience."""
    print("Conversational Data Lineage Agent with LLM Integration")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Lineage Agent CLI")
    parser.add_argument("--memory", choices=["buffer", "window"], default="buffer",
                        help="Type of memory to use (buffer=all history, window=recent turns)")
    parser.add_argument("--memory-db", type=str, default="lineage_memory.db", 
                        help="Path to SQLite database for persistent memory (default: lineage_memory.db)")
    parser.add_argument("--memory-window", type=int, default=5,
                        help="Number of conversation turns to remember when using window memory")
    parser.add_argument("--in-memory", action="store_true", 
                        help="Use in-memory storage instead of persistent storage")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List all available conversation sessions and exit")
    parser.add_argument("--session", type=str, default=None,
                        help="Load a specific session ID (use --list-sessions to see available sessions)")
    parser.add_argument("--no-human-feedback", action="store_true",
                        help="Disable human-in-the-loop feedback for uncertain responses")
    parser.add_argument("--interface", choices=["cli", "web"], default="cli",
                        help="Interface mode (cli for command line, web for web API)")
    args = parser.parse_args()
    
    # List sessions if requested
    if args.list_sessions:
        print("Available conversation sessions:")
        sessions = list_conversation_sessions(args.memory_db)
        if not sessions:
            print("  No sessions found")
        else:
            print("\n{:<5} {:<30} {:<20} {:<7} {:<40}".format(
                "No.", "Session ID", "Started", "Msgs", "Last Query"
            ))
            print("-" * 105)
            for i, (session_id, timestamp, msg_count, last_query) in enumerate(sessions, 1):
                print("{:<5} {:<30} {:<20} {:<7} {:<40}".format(
                    i, session_id, timestamp, msg_count, last_query or "N/A"
                ))
            print("\nUse --session SESSION_ID to continue a specific conversation")
        return
    
    # Handle the in-memory flag
    if args.in_memory:
        memory_db_path = None
        print("Using in-memory storage (conversations will not be saved)")
    else:
        memory_db_path = args.memory_db
        print(f"Using persistent memory database: {memory_db_path}")
    
    # Initialize the agent with specified memory settings
    agent = ConversationalLineageAgent(
        memory_type=args.memory,
        memory_db_path=memory_db_path,
        memory_k=args.memory_window,
        interface_mode=args.interface,
        enable_human_feedback=not args.no_human_feedback
    )
    
    # If a session ID was specified, override the generated one
    if args.session:
        agent.session_id = args.session
        print(f"Continuing conversation from session: {args.session}")
    
    print(f"Memory type: {args.memory}")
    if args.memory == "window":
        print(f"Memory window size: {args.memory_window} turns")
    
    print("\nEnter a query like 'trace customer_id' or 'show lineage for Customer Data Pipeline'")
    print("You can ask follow-up questions about previous results")
    
    # Show human feedback status
    if not args.no_human_feedback:
        print("\n✓ Human-in-the-loop feedback is enabled!")
        print("  When the agent is uncertain about a response, it will ask for your clarification.")
    else:
        print("\n✗ Human-in-the-loop feedback is disabled.")
        print("  Run with --no-human-feedback=false to enable feedback requests.")
    print("\nℹ️ Human-in-the-loop mode is enabled! If the agent is uncertain about a response,")
    print("   it will ask for your clarification to provide better answers.")

    while True:
        query = input("\nEnter your query (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            continue

        print("\nProcessing your request...")
        response = agent.process_message(query)
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)


# --- Entry Point ---
if __name__ == "__main__":
    main()
