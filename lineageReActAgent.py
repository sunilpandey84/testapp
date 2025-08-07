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
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

# Memory and ReAct imports
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

# Enhanced System Prompts with Memory Context
COORDINATOR_SYSTEM_PROMPT = """You are an intelligent Data Lineage Coordinator with conversation memory. Your role is to:

1. Analyze user requests for data lineage tracing while considering conversation history
2. Determine the most appropriate lineage strategy (contract-based or element-based)
3. Extract key parameters from natural language requests
4. Handle ambiguous or invalid requests by asking clarifying questions
5. Route the request to the appropriate specialized agent
6. Remember previous interactions and build upon them

CONVERSATION CONTEXT: {chat_history}

PREVIOUS ANALYSIS: {previous_analysis}

Guidelines:
- Reference previous conversations when relevant
- If user mentions a specific contract name, pipeline, or data flow, choose contract-based lineage
- If user mentions specific data elements, fields, or columns, choose element-based lineage  
- For invalid or unclear requests, provide helpful guidance with specific examples
- Always extract the traversal direction (upstream, downstream, or bidirectional) from context
- Be conversational and build rapport through memory of past interactions

Current request analysis: Determine the lineage type, extract parameters, and decide next steps.
"""

CONTRACT_SYSTEM_PROMPT = """You are an expert Contract Analysis Agent with memory capabilities. Your responsibilities:

1. Analyze data contracts and their associated pipelines
2. Remember previous contract analyses and build upon them
3. Identify all relevant ETL processes within a contract
4. Map dependencies between pipelines
5. Assess the complexity and scope of lineage tracing
6. Provide intelligent insights about the data flow

CONVERSATION CONTEXT: {chat_history}

PREVIOUS CONTRACT ANALYSES: {previous_contract_analyses}

Available tools: query_contract_by_name, query_pipelines_by_contract, query_pipeline_dependencies, query_element_mappings_by_queries

Guidelines:
- Reference previous contract analyses when relevant
- Compare current analysis with past findings
- Identify patterns across multiple contract analyses
- Suggest optimization opportunities based on historical data
"""

ELEMENT_SYSTEM_PROMPT = """You are an intelligent Element Tracing Agent with memory of previous traces. Your role is to:

1. Trace data element connections across the entire data ecosystem
2. Remember previously traced elements and their relationships
3. Understand transformation logic and business rules
4. Identify data quality and governance issues
5. Provide insights about data flow patterns
6. Detect circular dependencies or anomalies

CONVERSATION CONTEXT: {chat_history}

PREVIOUS ELEMENT TRACES: {previous_element_traces}

Available tools: find_element_by_name, trace_element_connections, query_element_mappings_by_queries

Guidelines:
- Build upon previously traced elements
- Identify patterns across multiple element traces
- Reference past findings when analyzing new elements
- Suggest related elements based on historical traces
"""

CONSTRUCTION_SYSTEM_PROMPT = """You are a Lineage Graph Construction Expert with memory of previous constructions. Your responsibilities:

1. Build comprehensive and accurate lineage graphs
2. Remember previous graph constructions and patterns
3. Optimize graph structure for clarity and usability
4. Add meaningful metadata and context
5. Identify patterns and insights in the lineage
6. Assess complexity and recommend visualization strategies

CONVERSATION CONTEXT: {chat_history}

PREVIOUS GRAPH CONSTRUCTIONS: {previous_constructions}

Guidelines:
- Build upon previous graph patterns
- Compare complexity with historical constructions
- Identify recurring patterns across different lineage graphs
- Suggest improvements based on past constructions
"""

APPROVAL_SYSTEM_PROMPT = """You are a Human Interaction Specialist with memory of previous interactions. Your role is to:

1. Process human feedback intelligently using conversation history
2. Handle ambiguous responses and natural language input
3. Validate user selections and descriptive inputs
4. Provide helpful guidance and explanations
5. Remember user preferences and communication style
6. Enable natural conversation flow

CONVERSATION CONTEXT: {chat_history}

USER INTERACTION HISTORY: {user_interaction_history}

Guidelines:
- Remember user's previous preferences and patterns
- Handle both structured selections (numbers) and natural language descriptions
- Parse complex user input and extract intent
- Provide personalized responses based on interaction history
- Be patient and adaptive to user communication style
"""


# Enhanced State with Memory
class LineageState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    chat_history: str
    user_interaction_history: Dict[str, Any]
    previous_analysis: Dict[str, Any]
    previous_contract_analyses: List[Dict[str, Any]]
    previous_element_traces: List[Dict[str, Any]]
    previous_constructions: List[Dict[str, Any]]

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

    # Enhanced Human-in-the-Loop (HITL) fields
    requires_human_approval: bool
    human_approval_message: Optional[str]
    human_feedback: Optional[Dict[str, Any]]
    human_input_type: str  # "selection", "description", "mixed"

    # Enhanced fields for LLM responses
    llm_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    complexity_score: Optional[int]

    # Memory and context fields
    conversation_memory: Any
    agent_memories: Dict[str, Any]
    invalid_request_count: int

    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]


class LineageRequest(BaseModel):
    """Enhanced request model to handle natural language inputs"""
    query: str = Field(description="Natural language query for lineage tracing")
    context: Optional[str] = Field(None, description="Additional context about the request")
    preferred_output: Optional[str] = Field(None, description="Preferred output format (graph, table, summary)")
    max_depth: Optional[int] = Field(5, description="Maximum depth for lineage tracing")
    user_id: Optional[str] = Field(None, description="User identifier for personalized memory")


# Enhanced Memory Manager
class ConversationMemoryManager:
    """Manages conversation memory across different agents"""

    def __init__(self, max_token_limit: int = 2000):
        self.memories = {}
        self.max_token_limit = max_token_limit

    def get_memory(self, agent_name: str, memory_type: str = "buffer"):
        """Get or create memory for an agent"""
        if agent_name not in self.memories:
            if memory_type == "summary":
                self.memories[agent_name] = ConversationSummaryBufferMemory(
                    llm=get_llm(),
                    max_token_limit=self.max_token_limit,
                    return_messages=True
                )
            else:
                self.memories[agent_name] = ConversationBufferWindowMemory(
                    k=10,  # Keep last 10 interactions
                    return_messages=True
                )
        return self.memories[agent_name]

    def add_interaction(self, agent_name: str, human_input: str, ai_response: str):
        """Add interaction to agent's memory"""
        memory = self.get_memory(agent_name)
        memory.chat_memory.add_user_message(human_input)
        memory.chat_memory.add_ai_message(ai_response)

    def get_conversation_history(self, agent_name: str) -> str:
        """Get formatted conversation history for an agent"""
        memory = self.get_memory(agent_name)
        try:
            history = memory.chat_memory.messages
            formatted_history = []
            for msg in history[-10:]:  # Last 10 messages
                if isinstance(msg, HumanMessage):
                    formatted_history.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    formatted_history.append(f"Assistant: {msg.content}")
            return "\n".join(formatted_history)
        except:
            return "No previous conversation history."


# Global memory manager
memory_manager = ConversationMemoryManager()


# Enhanced Database Manager (keeping original functionality)
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


# Enhanced tool functions (keeping original functionality)
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


# Enhanced Input Validation
class InputValidator:
    """Validates and processes user inputs"""

    @staticmethod
    def is_valid_lineage_query(query: str) -> Tuple[bool, str, List[str]]:
        """
        Validates if the query is a valid lineage request
        Returns: (is_valid, error_message, suggestions)
        """
        if not query or len(query.strip()) < 3:
            return False, "Query is too short. Please provide a more detailed request.", [
                "Try: 'trace customer_id lineage'",
                "Try: 'show Customer Data Pipeline'",
                "Try: 'analyze order_amount connections'"
            ]

        # Check for common invalid patterns
        invalid_patterns = [
            r'^(hi|hello|hey)$',
            r'^(what|how|when|where|why)\?*$',
            r'^(help|assistance|support)$',
            r'^(test|testing)$'
        ]

        query_lower = query.lower().strip()
        for pattern in invalid_patterns:
            if re.match(pattern, query_lower):
                return False, f"'{query}' is not a valid lineage query.", [
                    "Specify what you want to trace: 'trace [element_name]'",
                    "Ask about a contract: 'show [contract_name] pipeline'",
                    "Analyze relationships: 'analyze [element_name] connections'"
                ]

        # Check if query contains lineage-related keywords
        lineage_keywords = [
            'trace', 'lineage', 'pipeline', 'contract', 'element', 'connection',
            'flow', 'dependency', 'mapping', 'transformation', 'upstream', 'downstream'
        ]

        has_lineage_keyword = any(keyword in query_lower for keyword in lineage_keywords)

        if not has_lineage_keyword:
            return False, "Query doesn't seem to be related to data lineage.", [
                "Use keywords like: trace, lineage, pipeline, contract, element",
                "Example: 'trace customer_id lineage'",
                "Example: 'show Customer Data Pipeline dependencies'"
            ]

        return True, "", []

    @staticmethod
    def parse_human_feedback(feedback_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced parsing of human feedback supporting both selections and natural language
        """
        if not feedback_text:
            return {"type": "invalid", "message": "No input provided"}

        feedback_text = feedback_text.strip()

        # Try to parse as number selection first
        try:
            number = int(feedback_text)
            return {
                "type": "selection",
                "selected_index": number - 1,  # Convert to 0-based index
                "original_input": feedback_text
            }
        except ValueError:
            pass

        # Parse natural language descriptions
        feedback_lower = feedback_text.lower()

        # Extract intent from natural language
        if any(word in feedback_lower for word in ['customer', 'cust']):
            return {
                "type": "description",
                "intent": "customer_related",
                "keywords": ['customer', 'cust'],
                "original_input": feedback_text,
                "parsed_selection": feedback_text
            }
        elif any(word in feedback_lower for word in ['order', 'transaction']):
            return {
                "type": "description",
                "intent": "order_related",
                "keywords": ['order', 'transaction'],
                "original_input": feedback_text,
                "parsed_selection": feedback_text
            }
        elif any(word in feedback_lower for word in ['product', 'prod']):
            return {
                "type": "description",
                "intent": "product_related",
                "keywords": ['product', 'prod'],
                "original_input": feedback_text,
                "parsed_selection": feedback_text
            }

        # Check if it matches available options by name
        available_options = context.get("available_elements", []) or context.get("available_contracts", [])
        for i, option in enumerate(available_options):
            option_name = option.get("name", "").lower()
            if option_name and (feedback_lower in option_name or option_name in feedback_lower):
                return {
                    "type": "name_match",
                    "selected_index": i,
                    "matched_name": option.get("name"),
                    "original_input": feedback_text
                }

        # Generic descriptive input
        return {
            "type": "description",
            "intent": "general",
            "original_input": feedback_text,
            "parsed_selection": feedback_text
        }


# Enhanced Agent Functions with Memory
def initialize_state_memory(state: LineageState) -> LineageState:
    """Initialize memory components in state"""
    if not state.get("chat_history"):
        state["chat_history"] = ""
    if not state.get("user_interaction_history"):
        state["user_interaction_history"] = {"preferences": {}, "patterns": [], "invalid_attempts": 0}
    if not state.get("previous_analysis"):
        state["previous_analysis"] = {}
    if not state.get("previous_contract_analyses"):
        state["previous_contract_analyses"] = []
    if not state.get("previous_element_traces"):
        state["previous_element_traces"] = []
    if not state.get("previous_constructions"):
        state["previous_constructions"] = []
    if not state.get("agent_memories"):
        state["agent_memories"] = {}
    if not state.get("invalid_request_count"):
        state["invalid_request_count"] = 0
    return state


def lineage_coordinator_agent(state: LineageState):
    """Enhanced coordinator with memory and better invalid query handling."""
    logger.info("---EXECUTING: Enhanced Lineage Coordinator with Memory---")

    state = initialize_state_memory(state)

    llm = get_llm()
    user_query = state.get('input_parameter', '')

    # Update conversation memory
    chat_history = memory_manager.get_conversation_history("coordinator")
    state["chat_history"] = chat_history

    # Validate input first
    is_valid, error_msg, suggestions = InputValidator.is_valid_lineage_query(user_query)
    if not is_valid:
        state["invalid_request_count"] += 1
        return {
            **state,
            "requires_human_approval": True,
            "human_input_type": "description",
            "human_approval_message": f"""Invalid Query: {error_msg}

Here are some examples of valid lineage queries:
{chr(10).join([f'• {suggestion}' for suggestion in suggestions])}

Please provide a valid data lineage query, or ask me to:
1. List all available data elements
2. List all available contracts  
3. Show examples of lineage queries

What would you like to trace?""",
            "query_results": {"invalid_query": True, "suggestions": suggestions},
            "current_step": "coordinator"
        }

    # Get available data from database
    available_contracts = get_available_contracts.invoke({})
    available_elements = get_available_elements.invoke({})

    contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]
    element_names = [e['element_name'] for e in available_elements.get('elements', [])]

    # Enhanced context with memory
    context = f"""
    User Query: {user_query}

    Previous Analysis Summary: {state.get('previous_analysis', {})}

    Current State:
    - Messages: {len(state.get('messages', []))} messages in conversation
    - Invalid attempts: {state.get('invalid_request_count', 0)}
    - Previous context: {state.get('query_results', {})}

    Available Information in Database:
    - Data Contracts: {contract_names}
    - Data Elements: {element_names}

    Analysis Rules:
    - If query mentions specific element names -> element_based
    - If query mentions "contract", "pipeline" or contract names -> contract_based
    - If query asks to "trace [element_name]" -> element_based
    - Default traversal is bidirectional unless specified
    """

    # Format the system prompt with memory context
    formatted_prompt = COORDINATOR_SYSTEM_PROMPT.format(
        chat_history=chat_history,
        previous_analysis=json.dumps(state.get('previous_analysis', {}), indent=2)
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"{context}\n\nAnalyze this request and determine the lineage type and parameters.")
    ]

    try:
        response = llm.invoke(messages)

        # Update memory with this interaction
        memory_manager.add_interaction("coordinator", user_query, response.content)

        analysis = {
            "llm_reasoning": response.content,
            "confidence_score": 0.8,
            "timestamp": datetime.now().isoformat()
        }

        content = response.content.lower()
        user_query_lower = user_query.lower()

        # Enhanced element detection
        mentioned_element = None
        for element in element_names:
            if element.lower() in user_query_lower:
                mentioned_element = element
                break

        # Enhanced contract detection
        mentioned_contract = None
        for contract in contract_names:
            contract_words = contract.lower().split()
            if any(word in user_query_lower for word in contract_words):
                mentioned_contract = contract
                break

        # Determine lineage type
        if mentioned_element or (
                'trace' in user_query_lower and any(elem.lower() in user_query_lower for elem in element_names)):
            lineage_type = "element_based"
            element_name = mentioned_element

            if not element_name:
                import re
                match = re.search(r'trace\s+(\w+)', user_query_lower)
                if match:
                    potential_element = match.group(1)
                    for elem in element_names:
                        if elem.lower().startswith(potential_element) or potential_element in elem.lower():
                            element_name = elem
                            break

            if not element_name:
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_input_type": "mixed",
                    "human_approval_message": f"""I detected this might be an element-based query, but couldn't identify the specific element.

Available elements:
{chr(10).join([f'{i + 1}. {elem}' for i, elem in enumerate(element_names)])}

You can:
• Select a number (e.g., type '1' for {element_names[0] if element_names else 'first option'})
• Type the element name (e.g., 'customer_id')
• Describe what you're looking for (e.g., 'something related to customer information')

What element would you like to trace?""",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]},
                    "llm_analysis": analysis
                }

            # Search for element
            element_search_result = find_element_by_name.invoke({"element_name": element_name})
            if not element_search_result.get("success"):
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_input_type": "mixed",
                    "human_approval_message": f"""Could not find element '{element_name}' in the database.

Available elements:
{chr(10).join([f'{i + 1}. {elem}' for i, elem in enumerate(element_names)])}

Please:
• Select a number from the list above
• Type the exact element name
• Describe the type of data you're looking for

What element would you like to trace?""",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]},
                    "llm_analysis": analysis
                }

            found_elements = element_search_result["elements"]
            if len(found_elements) > 1:
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_input_type": "mixed",
                    "human_approval_message": f"""Multiple elements match '{element_name}':

{chr(10).join([f'{i + 1}. {elem["element_name"]} (in {elem["table_name"]})' for i, elem in enumerate(found_elements)])}

You can:
• Select a number (e.g., '1' for the first option)
• Describe which one you want (e.g., 'the one in customers table')

Which element do you want to trace?""",
                    "query_results": {"ambiguous_elements": found_elements},
                    "llm_analysis": analysis
                }

            state['element_queue'] = [found_elements[0]['element_code']]
            state['traced_elements'] = set()
            state['element_name'] = element_name
            next_step = "element_analysis"

        elif mentioned_contract or any(
                keyword in content or keyword in user_query_lower for keyword in ['contract', 'pipeline']):
            lineage_type = "contract_based"
            contract_name = mentioned_contract if mentioned_contract else user_query

            if not mentioned_contract:
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_input_type": "mixed",
                    "human_approval_message": f"""I detected this is a contract-based query.

Available contracts:
{chr(10).join([f'{i + 1}. {contract}' for i, contract in enumerate(contract_names)])}

You can:
• Select a number (e.g., '1' for {contract_names[0] if contract_names else 'first option'})
• Type the contract name
• Describe the pipeline you're interested in

Which contract would you like to analyze?""",
                    "query_results": {"available_contracts": [{"index": i, "name": contract} for i, contract in
                                                              enumerate(contract_names)]},
                    "llm_analysis": analysis
                }

            state['contract_name'] = contract_name
            next_step = "contract_analysis"
        else:
            return {
                **state,
                "requires_human_approval": True,
                "human_input_type": "description",
                "human_approval_message": f"""I need clarification about your request '{user_query}'.

Are you looking for:

**1. Element-based lineage** - Trace a specific data field
Available elements: {', '.join(element_names[:5])}{'...' if len(element_names) > 5 else ''}

**2. Contract-based lineage** - Analyze a data pipeline  
Available contracts: {', '.join(contract_names)}

Please tell me:
• Which type of analysis you want
• What specific element or contract to analyze
• Or describe what data you're trying to trace

What would you like to trace?""",
                "query_results": {
                    "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)],
                    "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)]
                },
                "llm_analysis": analysis
            }

        # Determine traversal direction
        if 'upstream' in content or 'upstream' in user_query_lower:
            direction = "upstream"
        elif 'downstream' in content or 'downstream' in user_query_lower:
            direction = "downstream"
        else:
            direction = "bidirectional"

        # Update previous analysis
        state["previous_analysis"] = analysis

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
    """Enhanced contract analysis with memory."""
    logger.info("---EXECUTING: Enhanced Contract Analysis with Memory---")

    llm = get_llm()
    contract_name = state.get('contract_name', state.get('input_parameter'))

    # Get conversation memory
    chat_history = memory_manager.get_conversation_history("contract_analysis")
    state["chat_history"] = chat_history

    # Gather contract data
    contract_details = query_contract_by_name.invoke({"contract_name": contract_name})
    if not contract_details.get("success"):
        return {**state, "error_message": contract_details.get("error"), "current_step": "error"}

    contract_code = contract_details["contract_code"]
    pipelines = query_pipelines_by_contract.invoke({"contract_code": contract_code})
    query_codes = [p['query_code'] for p in pipelines.get("pipelines", [])]
    dependencies = query_pipeline_dependencies.invoke({"query_codes": query_codes})
    mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})

    # Enhanced analysis context with memory
    analysis_context = f"""
    Contract Analysis with Historical Context:
    - Contract: {contract_details}
    - Pipelines: {pipelines.get("pipelines", [])}
    - Dependencies: {dependencies.get("dependencies", {})}
    - Element Mappings: {mappings.get("mappings", [])}
    - Previous Contract Analyses: {json.dumps(state.get("previous_contract_analyses", []), indent=2)}

    Please analyze this contract and provide:
    1. Key insights about the data flow
    2. Comparison with previous contract analyses
    3. Potential bottlenecks or issues
    4. Data quality considerations
    5. Recommendations for optimization
    6. Complexity assessment (1-10)
    """

    formatted_prompt = CONTRACT_SYSTEM_PROMPT.format(
        chat_history=chat_history,
        previous_contract_analyses=json.dumps(state.get("previous_contract_analyses", []), indent=2)
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=analysis_context)
    ]

    try:
        response = llm.invoke(messages)

        # Update memory
        memory_manager.add_interaction("contract_analysis", f"Analyze contract: {contract_name}", response.content)

        # Store this analysis in history
        current_analysis = {
            "contract_name": contract_name,
            "contract_code": contract_code,
            "analysis": response.content,
            "timestamp": datetime.now().isoformat(),
            "pipeline_count": len(pipelines.get("pipelines", [])),
            "mapping_count": len(mappings.get("mappings", []))
        }

        state["previous_contract_analyses"].append(current_analysis)

        complexity_score = min(len(query_codes) + len(mappings.get("mappings", [])), 10)
        recommendations = [
            "Monitor data quality in transformation steps",
            "Review pipeline dependencies for optimization",
            f"Contract has {len(query_codes)} pipelines - consider consolidation if needed"
        ]

        state["query_results"].update({
            "contract": contract_details,
            "pipelines": pipelines.get("pipelines", []),
            "dependencies": dependencies.get("dependencies", {}),
            "mappings": mappings.get("mappings", [])
        })

        state["llm_analysis"] = {
            "contract_insights": response.content,
            "complexity_assessment": complexity_score,
            "historical_comparison": f"This is analysis #{len(state['previous_contract_analyses'])}"
        }
        state["recommendations"] = recommendations
        state["complexity_score"] = complexity_score

        return {**state, "current_step": "lineage_construction"}

    except Exception as e:
        logger.error(f"Error in contract analysis: {e}")
        return {**state, "error_message": f"Contract analysis failed: {str(e)}", "current_step": "error"}


def element_analysis_agent(state: LineageState):
    """Enhanced element analysis with memory."""
    logger.info("---EXECUTING: Enhanced Element Analysis with Memory---")

    llm = get_llm()

    # Get conversation memory
    chat_history = memory_manager.get_conversation_history("element_analysis")
    state["chat_history"] = chat_history

    if not state.get("element_queue"):
        return {**state, "current_step": "lineage_construction"}

    element_to_trace = state["element_queue"].pop(0)

    if element_to_trace in state["traced_elements"]:
        return {**state}

    state["traced_elements"].add(element_to_trace)

    # Get all available query codes
    all_query_codes_result = get_all_query_codes.invoke({})
    query_codes = all_query_codes_result.get("query_codes", ["Q001", "Q002", "Q003"])

    # Get element mappings
    all_mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})
    related_mappings = [m for m in all_mappings.get("mappings", [])
                        if m['source_code'] == element_to_trace or m['target_code'] == element_to_trace]

    # Enhanced element context with memory
    element_context = f"""
    Element Analysis with Historical Context:
    - Current Element: {element_to_trace}
    - Related Mappings: {json.dumps(related_mappings, indent=2)}
    - Previous Element Traces: {json.dumps(state.get("previous_element_traces", []), indent=2)}
    - Previously Traced Elements: {list(state.get("traced_elements", set()))}

    Please analyze:
    1. Data transformation patterns
    2. Comparison with previously traced elements
    3. Business logic implications
    4. Data quality risks
    5. Governance considerations
    6. Suggest next elements to trace
    """

    formatted_prompt = ELEMENT_SYSTEM_PROMPT.format(
        chat_history=chat_history,
        previous_element_traces=json.dumps(state.get("previous_element_traces", []), indent=2)
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=element_context)
    ]

    try:
        response = llm.invoke(messages)

        # Update memory
        memory_manager.add_interaction("element_analysis", f"Trace element: {element_to_trace}", response.content)

        # Store this trace in history
        current_trace = {
            "element_code": element_to_trace,
            "analysis": response.content,
            "timestamp": datetime.now().isoformat(),
            "related_mappings_count": len(related_mappings)
        }

        state["previous_element_traces"].append(current_trace)

        # Initialize mappings list if it doesn't exist
        if "mappings" not in state["query_results"]:
            state["query_results"]["mappings"] = []

        # Add only new mappings to avoid duplication
        existing_mappings = state["query_results"]["mappings"]
        existing_mapping_keys = {(em['query_code'], em['source_code'], em['target_code']) for em in existing_mappings}

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
    """Enhanced lineage construction with memory."""
    logger.info("---EXECUTING: Enhanced Lineage Construction with Memory---")

    llm = get_llm()

    # Get conversation memory
    chat_history = memory_manager.get_conversation_history("construction")
    state["chat_history"] = chat_history

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

        # Add edges
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

    # Enhanced graph context with memory
    graph_context = f"""
    Lineage Graph Analysis with Historical Context:
    - Total Nodes: {len(nodes)}
    - Total Edges: {len(edges)}
    - Previous Constructions: {json.dumps(state.get("previous_constructions", []), indent=2)}
    - Node Details: {json.dumps(nodes, indent=2)[:1000]}...
    - Edge Details: {json.dumps(edges, indent=2)[:1000]}...

    Please analyze this lineage graph and provide:
    1. Graph complexity assessment (1-10)
    2. Comparison with previous constructions
    3. Critical path identification
    4. Transformation complexity analysis
    5. Recommendations for graph optimization
    6. Potential data governance issues
    """

    formatted_prompt = CONSTRUCTION_SYSTEM_PROMPT.format(
        chat_history=chat_history,
        previous_constructions=json.dumps(state.get("previous_constructions", []), indent=2)
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=graph_context)
    ]

    try:
        response = llm.invoke(messages)

        # Update memory
        memory_manager.add_interaction("construction", f"Construct graph with {len(nodes)} nodes", response.content)

        # Store this construction in history
        current_construction = {
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "analysis": response.content,
            "timestamp": datetime.now().isoformat(),
            "lineage_type": state.get("lineage_type")
        }

        state["previous_constructions"].append(current_construction)

        # Update state
        state["lineage_nodes"] = nodes
        state["lineage_edges"] = edges

        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["graph_construction"] = response.content

        complexity_score = min((len(edges) + len(nodes)) // 2, 10)
        state["complexity_score"] = complexity_score

        return {**state, "current_step": "finalize_results"}

    except Exception as e:
        logger.error(f"Error in lineage construction: {e}")
        return {**state, "error_message": f"Graph construction failed: {str(e)}", "current_step": "error"}


def human_approval_agent(state: LineageState):
    """Enhanced human interaction with natural language support."""
    logger.info("---EXECUTING: Enhanced Human Approval Agent---")

    # Get conversation memory
    chat_history = memory_manager.get_conversation_history("human_approval")
    state["chat_history"] = chat_history

    if not state.get("requires_human_approval"):
        return {**state, "current_step": "error"}

    human_feedback = state.get("human_feedback")
    if not human_feedback:
        logger.info("Waiting for human feedback...")
        return state

    logger.info(f"Processing human feedback: {human_feedback}")

    try:
        # Enhanced feedback processing
        query_results = state.get("query_results", {})

        # Handle invalid query responses
        if query_results.get("invalid_query"):
            feedback_text = human_feedback.get("description") or human_feedback.get("original_input", "")

            # Check if the new input is a valid lineage query
            is_valid, error_msg, suggestions = InputValidator.is_valid_lineage_query(feedback_text)

            if is_valid:
                # Restart the process with the new valid query
                return {
                    **state,
                    "input_parameter": feedback_text,
                    "requires_human_approval": False,
                    "human_feedback": None,
                    "current_step": "coordinator",
                    "query_results": {}  # Clear previous invalid query results
                }
            else:
                # Still invalid, ask again
                state["invalid_request_count"] += 1
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"""Still not a valid lineage query: {error_msg}

Attempt {state['invalid_request_count']} of 3.

{chr(10).join([f'• {suggestion}' for suggestion in suggestions])}

Please provide a specific data lineage request:""",
                    "human_feedback": None
                }

        # Parse the feedback using enhanced parser
        parsed_feedback = InputValidator.parse_human_feedback(
            human_feedback.get("description") or human_feedback.get("original_input", ""),
            query_results
        )

        # Handle different types of feedback
        if parsed_feedback["type"] == "selection":
            selected_index = parsed_feedback["selected_index"]

            # Handle element disambiguation
            if "ambiguous_elements" in query_results:
                elements = query_results["ambiguous_elements"]
                if 0 <= selected_index < len(elements):
                    selected_element = elements[selected_index]
                    memory_manager.add_interaction("human_approval",
                                                   f"Selected element {selected_index + 1}",
                                                   f"Proceeding with {selected_element['element_name']}")

                    return {
                        **state,
                        "element_queue": [selected_element["element_code"]],
                        "traced_elements": set(),
                        "element_name": selected_element["element_name"],
                        "lineage_type": "element_based",
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "element_analysis"
                    }

            # Handle element selection from available options
            elif "available_elements" in query_results:
                available_elements = query_results["available_elements"]
                if 0 <= selected_index < len(available_elements):
                    selected_element_name = available_elements[selected_index]["name"]
                    element_search_result = find_element_by_name.invoke({"element_name": selected_element_name})

                    if element_search_result.get("success") and element_search_result["elements"]:
                        found_elements = element_search_result["elements"]
                        memory_manager.add_interaction("human_approval",
                                                       f"Selected element: {selected_element_name}",
                                                       "Element found and analysis starting")

                        return {
                            **state,
                            "element_queue": [found_elements[0]["element_code"]],
                            "traced_elements": set(),
                            "element_name": selected_element_name,
                            "lineage_type": "element_based",
                            "requires_human_approval": False,
                            "human_feedback": None,
                            "current_step": "element_analysis"
                        }

            # Handle contract selection
            elif "available_contracts" in query_results:
                available_contracts = query_results["available_contracts"]
                if 0 <= selected_index < len(available_contracts):
                    selected_contract_name = available_contracts[selected_index]["name"]
                    memory_manager.add_interaction("human_approval",
                                                   f"Selected contract: {selected_contract_name}",
                                                   "Contract analysis starting")

                    return {
                        **state,
                        "contract_name": selected_contract_name,
                        "lineage_type": "contract_based",
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "contract_analysis"
                    }

        elif parsed_feedback["type"] in ["description", "name_match"]:
            # Handle natural language descriptions
            feedback_text = parsed_feedback["original_input"]

            # Try to match with available options
            if "available_elements" in query_results:
                elements = query_results["available_elements"]

                # Look for matches based on description
                matched_elements = []
                for i, elem in enumerate(elements):
                    elem_name = elem["name"].lower()
                    if (feedback_text.lower() in elem_name or
                            elem_name in feedback_text.lower() or
                            any(keyword in elem_name for keyword in parsed_feedback.get("keywords", []))):
                        matched_elements.append((i, elem))

                if len(matched_elements) == 1:
                    # Exact match found
                    selected_index, selected_elem = matched_elements[0]
                    element_search_result = find_element_by_name.invoke({"element_name": selected_elem["name"]})

                    if element_search_result.get("success") and element_search_result["elements"]:
                        found_elements = element_search_result["elements"]
                        memory_manager.add_interaction("human_approval",
                                                       f"Matched description to: {selected_elem['name']}",
                                                       "Starting element analysis")

                        return {
                            **state,
                            "element_queue": [found_elements[0]["element_code"]],
                            "traced_elements": set(),
                            "element_name": selected_elem["name"],
                            "lineage_type": "element_based",
                            "requires_human_approval": False,
                            "human_feedback": None,
                            "current_step": "element_analysis"
                        }

                elif len(matched_elements) > 1:
                    # Multiple matches - ask for clarification
                    return {
                        **state,
                        "requires_human_approval": True,
                        "human_input_type": "mixed",
                        "human_approval_message": f"""Your description "{feedback_text}" matches multiple elements:

{chr(10).join([f'{i + 1}. {elem["name"]}' for i, (_, elem) in enumerate(matched_elements)])}

Please:
• Select a number (e.g., '1' for the first option)
• Be more specific in your description

Which element do you want?""",
                        "query_results": {
                            "ambiguous_elements": [{"element_code": f"DE{i:03d}", "element_name": elem["name"]} for
                                                   i, (_, elem) in enumerate(matched_elements)]},
                        "human_feedback": None
                    }

            # Handle contract descriptions similarly
            elif "available_contracts" in query_results:
                contracts = query_results["available_contracts"]

                # Look for matches based on description
                matched_contracts = []
                for i, contract in enumerate(contracts):
                    contract_name = contract["name"].lower()
                    if (feedback_text.lower() in contract_name or
                            contract_name in feedback_text.lower()):
                        matched_contracts.append((i, contract))

                if len(matched_contracts) == 1:
                    # Exact match found
                    selected_index, selected_contract = matched_contracts[0]
                    memory_manager.add_interaction("human_approval",
                                                   f"Matched description to: {selected_contract['name']}",
                                                   "Starting contract analysis")

                    return {
                        **state,
                        "contract_name": selected_contract["name"],
                        "lineage_type": "contract_based",
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "contract_analysis"
                    }

        # If we reach here, the feedback couldn't be processed successfully
        logger.warning("Could not process human feedback successfully")
        return {
            **state,
            "requires_human_approval": True,
            "human_input_type": "mixed",
            "human_approval_message": f"""I couldn't understand your input "{parsed_feedback.get('original_input', '')}'.

Please try:
• Selecting a number from the available options
• Typing the exact name of what you want
• Being more specific in your description

{state.get('human_approval_message', 'Please provide your selection:')}""",
            "human_feedback": None
        }

    except Exception as e:
        logger.error(f"Error processing human feedback: {e}")
        traceback.print_exc()
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": f"Error processing your input: {str(e)}\n\nPlease try again with a clear selection or description.",
            "human_feedback": None
        }


def finalize_results_agent(state: LineageState):
    """Enhanced result finalization with memory context."""
    logger.info("---EXECUTING: Enhanced Result Finalization---")

    llm = get_llm()

    # Get conversation memory
    chat_history = memory_manager.get_conversation_history("finalize")

    # Deduplicate final results
    def deduplicate_mappings(mappings):
        seen = set()
        unique_mappings = []
        for mapping in mappings:
            key = (mapping['query_code'], mapping['source_code'], mapping['target_code'])
            if key not in seen:
                unique_mappings.append(mapping)
                seen.add(key)
        return unique_mappings

    def deduplicate_edges(edges):
        seen = set()
        unique_edges = []
        for edge in edges:
            key = (edge['source'], edge['target'], edge['query_code'])
            if key not in seen:
                unique_edges.append(edge)
                seen.add(key)
        return unique_edges

    final_mappings = deduplicate_mappings(state.get("query_results", {}).get("mappings", []))
    final_edges = deduplicate_edges(state.get("lineage_edges", []))

    # Update state with cleaned data
    if "query_results" not in state:
        state["query_results"] = {}
    state["query_results"]["mappings"] = final_mappings
    state["lineage_edges"] = final_edges

    # Compile comprehensive results with memory context
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
        "memory_context": {
            "conversation_history": chat_history,
            "total_interactions": len(state.get("previous_analysis", {})),
            "previous_contract_analyses_count": len(state.get("previous_contract_analyses", [])),
            "previous_element_traces_count": len(state.get("previous_element_traces", [])),
            "previous_constructions_count": len(state.get("previous_constructions", []))
        },
        "timestamp": datetime.now().isoformat()
    }

    # Generate enhanced executive summary with memory context
    summary_context = f"""
    Comprehensive Lineage Analysis Complete:

    Current Analysis:
    - Analysis Type: {state.get("lineage_type")}
    - Input: {state.get("input_parameter")}
    - Nodes Found: {len(state.get("lineage_nodes", []))}
    - Relationships: {len(final_edges)}
    - Complexity Score: {state.get("complexity_score", 0)}/10

    Historical Context:
    - Previous Contract Analyses: {len(state.get("previous_contract_analyses", []))}
    - Previous Element Traces: {len(state.get("previous_element_traces", []))}
    - Previous Graph Constructions: {len(state.get("previous_constructions", []))}
    - Conversation History: {chat_history[-500:]}...

    LLM Insights: {json.dumps(state.get("llm_analysis", {}), indent=2)[:1000]}...

    Generate a comprehensive executive summary including:
    1. Key findings and insights from current analysis
    2. How this analysis relates to previous work
    3. Data flow patterns discovered
    4. Risk assessment and recommendations
    5. Next steps for data governance
    6. Lessons learned from conversation history
    """

    messages = [
        SystemMessage(
            content="You are an expert data analyst creating executive summaries of lineage analysis results with historical context."),
        HumanMessage(content=summary_context)
    ]

    try:
        response = llm.invoke(messages)
        final_result["executive_summary"] = response.content

        # Update memory with final results
        memory_manager.add_interaction("finalize",
                                       f"Complete analysis: {state.get('lineage_type')} for {state.get('input_parameter')}",
                                       f"Analysis completed with {len(final_edges)} relationships found")

        return {**state, "final_result": final_result, "current_step": "complete"}

    except Exception as e:
        logger.error(f"Error in result finalization: {e}")
        final_result["executive_summary"] = "Summary generation failed, but analysis completed successfully."
        return {**state, "final_result": final_result, "current_step": "complete"}


def error_handler_agent(state: LineageState):
    """Enhanced error handling with memory and recovery suggestions."""
    logger.info("---EXECUTING: Enhanced Error Handler---")

    error_msg = state.get("error_message", "Unknown error occurred")
    chat_history = memory_manager.get_conversation_history("error_handler")

    # Enhanced error response with context
    error_result = {
        "success": False,
        "error": error_msg,
        "suggestions": [
            "Check if the input parameter exists in the database",
            "Try using more specific search terms",
            "Verify database connectivity and schema",
            "Review conversation history for similar successful queries"
        ],
        "memory_context": {
            "conversation_history": chat_history,
            "previous_successful_queries": [
                analysis.get("input_parameter") for analysis in state.get("previous_analysis", {}).values()
                if isinstance(analysis, dict) and analysis.get("input_parameter")
            ]
        },
        "timestamp": datetime.now().isoformat()
    }

    # Update memory with error
    memory_manager.add_interaction("error_handler", f"Error occurred: {error_msg}",
                                   "Error logged and suggestions provided")

    return {**state, "final_result": error_result, "current_step": "complete"}


# Enhanced Route Decision Functions
def should_continue(state: LineageState):
    """Enhanced routing with memory awareness."""
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
    """Enhanced element continuation with memory."""
    if state.get("element_queue") and len(state.get("element_queue", [])) > 0:
        return "element_analysis"
    else:
        return "lineage_construction"


# ReAct Agent Integration
def create_lineage_react_agent():
    """Create a ReAct agent for complex lineage queries that need tool usage."""

    tools = [
        query_contract_by_name,
        query_pipelines_by_contract,
        query_pipeline_dependencies,
        query_element_mappings_by_queries,
        find_element_by_name,
        trace_element_connections,
        get_all_query_codes,
        get_available_contracts,
        get_available_elements
    ]

    react_prompt = PromptTemplate.from_template("""
You are a data lineage expert agent. Use the available tools to answer questions about data lineage, contracts, and element relationships.

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation context: {chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")

    llm = get_llm()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )


def react_agent_wrapper(state: LineageState):
    """Wrapper to use ReAct agent for complex queries."""
    logger.info("---EXECUTING: ReAct Agent for Complex Query---")

    try:
        react_agent = create_lineage_react_agent()

        user_query = state.get('input_parameter', '')
        chat_history = state.get('chat_history', '')

        # Use ReAct agent for complex queries
        result = react_agent.invoke({
            "input": user_query,
            "chat_history": chat_history
        })

        # Process ReAct result and convert to lineage format
        react_output = result.get("output", "")

        # Store ReAct analysis
        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["react_analysis"] = react_output

        return {
            **state,
            "current_step": "finalize_results",
            "final_result": {
                "react_analysis": react_output,
                "timestamp": datetime.now().isoformat(),
                "lineage_type": "react_powered",
                "input_parameter": user_query
            }
        }

    except Exception as e:
        logger.error(f"Error in ReAct agent: {e}")
        return {
            **state,
            "error_message": f"ReAct agent failed: {str(e)}",
            "current_step": "error"
        }


# Enhanced Lineage Orchestrator with ReAct Integration
class EnhancedLineageOrchestrator:
    """Enhanced orchestrator with memory, ReAct agents, and improved human interaction."""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.memory_manager = memory_manager
        self.graph = self._build_enhanced_workflow()
        self.react_agent = create_lineage_react_agent()
        self.paused_state = None
        self.workflow_config = None

    def _build_enhanced_workflow(self):
        """Build the enhanced LangGraph workflow with ReAct integration."""
        workflow = StateGraph(LineageState)

        # Add all enhanced agent nodes
        workflow.add_node("coordinator", lineage_coordinator_agent)
        workflow.add_node("contract_analysis", contract_analysis_agent)
        workflow.add_node("element_analysis", element_analysis_agent)
        workflow.add_node("lineage_construction", lineage_construction_agent)
        workflow.add_node("human_approval", human_approval_agent)
        workflow.add_node("finalize_results", finalize_results_agent)
        workflow.add_node("error_handler", error_handler_agent)
        workflow.add_node("react_agent", react_agent_wrapper)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional routing
        workflow.add_conditional_edges("coordinator", should_continue)
        workflow.add_conditional_edges("contract_analysis", should_continue)
        workflow.add_conditional_edges("element_analysis", element_continue_condition)
        workflow.add_conditional_edges("lineage_construction", should_continue)
        workflow.add_conditional_edges("human_approval", should_continue)
        workflow.add_conditional_edges("finalize_results", should_continue)
        workflow.add_conditional_edges("react_agent", should_continue)
        workflow.add_edge("error_handler", END)

        # Compile with interrupt before human_approval
        return workflow.compile(interrupt_before=["human_approval"])

    def should_use_react_agent(self, query: str) -> bool:
        """
        Determine if the query is complex enough to warrant ReAct agent usage.
        ReAct is recommended for:
        1. Multi-step reasoning queries
        2. Queries requiring multiple tool calls
        3. Complex analytical questions
        """
        complex_patterns = [
            r'compare.*between.*and',
            r'analyze.*and.*show',
            r'find.*then.*trace',
            r'what.*if.*then',
            r'multiple.*pipeline',
            r'all.*contract.*with',
            r'relationship.*between.*and'
        ]

        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in complex_patterns)

    async def execute_lineage_request(self, request: LineageRequest) -> Dict[str, Any]:
        """Enhanced execution with ReAct agent decision making."""

        # Reset state
        self.paused_state = None
        self.workflow_config = None

        try:
            # Check if we should use ReAct agent
            use_react = self.should_use_react_agent(request.query)

            if use_react:
                logger.info("Using ReAct agent for complex query")
                # Use ReAct agent directly for complex queries
                try:
                    react_result = self.react_agent.invoke({
                        "input": request.query,
                        "chat_history": self.memory_manager.get_conversation_history("react")
                    })

                    return {
                        "success": True,
                        "analysis_type": "react_powered",
                        "query": request.query,
                        "result": react_result.get("output", ""),
                        "recommendation": "Used ReAct agent for complex multi-step reasoning",
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"ReAct agent failed, falling back to standard workflow: {e}")
                    # Fall back to standard workflow

            # Standard workflow execution
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
                "human_input_type": "mixed",
                "llm_analysis": {},
                "recommendations": [],
                "complexity_score": None,
                "chat_history": "",
                "user_interaction_history": {"preferences": {}, "patterns": [], "invalid_attempts": 0},
                "previous_analysis": {},
                "previous_contract_analyses": [],
                "previous_element_traces": [],
                "previous_constructions": [],
                "agent_memories": {},
                "invalid_request_count": 0,
                "conversation_memory": None,
                "final_result": None,
                "error_message": None
            }

            # Create unique thread ID
            thread_id = f"thread_{id(request)}_{datetime.now().timestamp()}"
            config = {"configurable": {"thread_id": thread_id}}
            self.workflow_config = config

            final_state = None
            async for event in self.graph.astream(initial_state, config):
                logger.info(f"Processing event: {list(event.keys())}")

                for node_name, state_data in event.items():
                    if isinstance(state_data, tuple) and len(state_data) > 1:
                        state_data = state_data[1] if isinstance(state_data[1], dict) else state_data[0]

                    if not isinstance(state_data, dict):
                        continue

                    final_state = state_data

                    # Check if human approval is required
                    if state_data.get("requires_human_approval"):
                        logger.info("---WORKFLOW PAUSED FOR HUMAN INPUT---")
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "input_type": state_data.get("human_input_type", "mixed"),
                            "message": state_data.get("human_approval_message", "Human input required."),
                            "query_results": state_data.get("query_results", {}),
                            "thread_id": thread_id,
                            "supports_natural_language": True,
                            "examples": [
                                "You can type a number (e.g., '1')",
                                "Or describe what you want (e.g., 'something related to customers')",
                                "Or type the exact name of what you're looking for"
                            ]
                        }

            # Workflow completed
            if final_state and final_state.get("final_result"):
                result = final_state["final_result"]
                result[
                    "react_recommendation"] = "Standard workflow completed successfully. Consider ReAct agent for more complex multi-step queries."
                return result
            else:
                return {"error": "Workflow completed but no final result generated"}

        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"Enhanced workflow execution failed: {str(e)}"}

    async def resume_with_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced feedback resumption supporting natural language."""
        if not self.paused_state or not self.workflow_config:
            return {"error": "No paused workflow to resume. Please start a new request."}

        try:
            logger.info(f"Resuming enhanced workflow with feedback: {feedback}")

            # Enhanced feedback processing - handle both structured and unstructured input
            processed_feedback = feedback

            # If feedback is just a string, wrap it in proper structure
            if isinstance(feedback, str):
                processed_feedback = {"description": feedback, "original_input": feedback}
            elif "text" in feedback and not "description" in feedback:
                processed_feedback = {"description": feedback["text"], "original_input": feedback["text"]}

            updated_state = {**self.paused_state}
            updated_state["human_feedback"] = processed_feedback
            updated_state["requires_human_approval"] = True

            final_state = None
            async for event in self.graph.astream(updated_state, self.workflow_config):
                logger.info(f"Resume event: {list(event.keys())}")

                for node_name, state_data in event.items():
                    if isinstance(state_data, tuple) and len(state_data) > 1:
                        state_data = state_data[1] if isinstance(state_data[1], dict) else state_data[0]

                    if not isinstance(state_data, dict):
                        continue

                    final_state = state_data

                    # Check if another human approval is needed
                    if (state_data.get("requires_human_approval") and
                            state_data.get("human_feedback") is None):
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "input_type": state_data.get("human_input_type", "mixed"),
                            "message": state_data.get("human_approval_message", "Additional input required."),
                            "query_results": state_data.get("query_results", {}),
                            "supports_natural_language": True
                        }

            # Clear paused state after completion
            self.paused_state = None
            self.workflow_config = None

            if final_state and final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return {"error": "Enhanced workflow resumed but no final result generated"}

        except Exception as e:
            logger.error(f"Failed to resume enhanced workflow: {e}")
            traceback.print_exc()
            return {"error": f"Failed to resume enhanced workflow: {str(e)}"}


# Enhanced main function with better interaction
async def main():
    """Enhanced main function with improved user experience."""
    orchestrator = EnhancedLineageOrchestrator()

    print("🚀 Enhanced Data Lineage Agent with Memory & ReAct Integration")
    print("=" * 70)
    print("Features:")
    print("• Remembers conversation history")
    print("• Supports natural language feedback")
    print("• ReAct agent for complex queries")
    print("• Enhanced error handling")
    print("=" * 70)
    print("\nExample queries:")
    print("• 'trace customer_id lineage'")
    print("• 'show Customer Data Pipeline'")
    print("• 'compare contracts and show dependencies'")  # ReAct example
    print("• 'find all elements related to customers then trace them'")  # ReAct example

    while True:
        query = input("\n💬 Enter your lineage query (or 'exit'): ").strip()
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        if not query:
            continue

        request = LineageRequest(query=query)
        print("\n🔍 Processing your request...")

        # Check if ReAct agent would be recommended
        if orchestrator.should_use_react_agent(query):
            print("🤖 This looks like a complex query - using ReAct agent for better reasoning")

        result = await orchestrator.execute_lineage_request(request)

        # Handle human feedback loop
        while result.get("human_input_required"):
            print("\n" + "=" * 50)
            print("🙋 HUMAN INPUT NEEDED")
            print("=" * 50)
            print(result.get("message"))

            if result.get("supports_natural_language"):
                print("\n💡 You can respond in multiple ways:")
                for example in result.get("examples", []):
                    print(f"   {example}")

            print("=" * 50)

            user_response = input("Your response: ").strip()

            print("\n⏳ Processing your response...")
            result = await orchestrator.resume_with_feedback(user_response)

        # Display final results
        print("\n" + "=" * 70)
        print("📊 LINEAGE ANALYSIS RESULTS")
        print("=" * 70)

        if result.get("success", True):
            print("✅ Analysis completed successfully!")
            if "react_recommendation" in result:
                print(f"💡 {result['react_recommendation']}")

        print(json.dumps(result, indent=2, default=str))


# Entry Point
if __name__ == "__main__":
    asyncio.run(main())