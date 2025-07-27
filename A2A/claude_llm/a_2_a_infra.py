"""
Agent-to-Agent (A2A) Communication Framework
Base infrastructure for independent agents with message passing
"""
import sqlite3
from langchain_core.tools import tool
import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
import traceback
from collections import defaultdict
import os

from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be passed between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_ERROR = "task_error"
    HUMAN_INPUT_REQUIRED = "human_input_required"
    HUMAN_FEEDBACK = "human_feedback"
    BROADCAST = "broadcast"
    SYSTEM = "system"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Core message structure for A2A communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    sender_id: str = ""
    recipient_id: str = ""
    task_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For tracking related messages
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=MessageType(data.get('type', MessageType.TASK_REQUEST.value)),
            priority=MessagePriority(data.get('priority', MessagePriority.NORMAL.value)),
            sender_id=data.get('sender_id', ''),
            recipient_id=data.get('recipient_id', ''),
            task_type=data.get('task_type', ''),
            payload=data.get('payload', {}),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            correlation_id=data.get('correlation_id'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            timeout=data.get('timeout')
        )


class MessageBus:
    """Central message bus for agent communication with routing and delivery"""

    def __init__(self):
        self.agents: Dict[str, 'BaseA2AAgent'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[Message] = []
        self.routing_rules: Dict[str, List[str]] = defaultdict(list)
        self.running = False
        self.bus_task: Optional[asyncio.Task] = None

    def register_agent(self, agent: 'BaseA2AAgent'):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        agent.set_message_bus(self)
        logger.info(f"Registered agent: {agent.agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def add_routing_rule(self, task_type: str, agent_ids: List[str]):
        """Add routing rule for specific task types"""
        self.routing_rules[task_type] = agent_ids

    async def send_message(self, message: Message):
        """Send a message through the bus"""
        message.timestamp = datetime.now()
        await self.message_queue.put(message)
        self.message_history.append(message)
        logger.debug(f"Message queued: {message.id} from {message.sender_id} to {message.recipient_id}")

    async def broadcast_message(self, message: Message, exclude_sender: bool = True):
        """Broadcast message to all registered agents"""
        message.type = MessageType.BROADCAST
        recipients = list(self.agents.keys())
        if exclude_sender and message.sender_id in recipients:
            recipients.remove(message.sender_id)

        for agent_id in recipients:
            msg_copy = Message(
                type=message.type,
                sender_id=message.sender_id,
                recipient_id=agent_id,
                task_type=message.task_type,
                payload=message.payload.copy(),
                metadata=message.metadata.copy(),
                correlation_id=message.correlation_id
            )
            await self.send_message(msg_copy)

    async def start(self):
        """Start the message bus"""
        if self.running:
            return

        self.running = True
        self.bus_task = asyncio.create_task(self._message_processor())
        logger.info("Message bus started")

    async def stop(self):
        """Stop the message bus"""
        if not self.running:
            return

        self.running = False
        if self.bus_task:
            self.bus_task.cancel()
            try:
                await self.bus_task
            except asyncio.CancelledError:
                pass
        logger.info("Message bus stopped")

    async def _message_processor(self):
        """Main message processing loop"""
        while self.running:
            try:
                # Wait for message with timeout to allow for clean shutdown
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._deliver_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()

    async def _deliver_message(self, message: Message):
        """Deliver message to appropriate agent(s)"""
        try:
            # Handle specific recipient
            if message.recipient_id and message.recipient_id in self.agents:
                agent = self.agents[message.recipient_id]
                await agent.receive_message(message)
                return

            # Handle task-based routing
            if message.task_type in self.routing_rules:
                for agent_id in self.routing_rules[message.task_type]:
                    if agent_id in self.agents:
                        agent = self.agents[agent_id]
                        await agent.receive_message(message)
                return

            # Handle broadcast messages
            if message.type == MessageType.BROADCAST:
                for agent_id, agent in self.agents.items():
                    if agent_id != message.sender_id:
                        await agent.receive_message(message)
                return

            logger.warning(f"No route found for message: {message.id}")

        except Exception as e:
            logger.error(f"Error delivering message {message.id}: {e}")
            # Send error message back to sender if possible
            if message.sender_id in self.agents:
                error_msg = Message(
                    type=MessageType.TASK_ERROR,
                    sender_id="message_bus",
                    recipient_id=message.sender_id,
                    correlation_id=message.id,
                    payload={'error': str(e), 'original_message': message.to_dict()}
                )
                await self.send_message(error_msg)

    async def add_default_routing(self):
        """Add default routing rules for lineage analysis"""
        # Route task responses back to client
        self.add_routing_rule("finalize_results", ["client_response_handler"])
        self.add_routing_rule("handle_error", ["client_response_handler"])

        # Route human input requests
        self.add_routing_rule("human_input_required", ["human_approval_agent"])


class BaseA2AAgent(ABC):
    """Base class for all A2A agents"""

    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.message_bus: Optional[MessageBus] = None
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.agent_task: Optional[asyncio.Task] = None
        self.handlers: Dict[str, Callable] = {}
        self.context: Dict[str, Any] = {}
        self.metrics = {
            'messages_processed': 0,
            'messages_sent': 0,
            'errors': 0,
            'start_time': None
        }

        # Register default handlers
        self._register_default_handlers()

    def set_message_bus(self, bus: MessageBus):
        """Set the message bus for this agent"""
        self.message_bus = bus

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.handlers['system'] = self._handle_system_message
        self.handlers['task_error'] = self._handle_error_message
        self.handlers['broadcast'] = self._handle_broadcast_message

    def register_handler(self, task_type: str, handler: Callable):
        """Register a custom message handler"""
        self.handlers[task_type] = handler

    async def start(self):
        """Start the agent"""
        if self.running:
            return

        self.running = True
        self.metrics['start_time'] = datetime.now()
        self.agent_task = asyncio.create_task(self._message_loop())
        logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent"""
        if not self.running:
            return

        self.running = False
        if self.agent_task:
            self.agent_task.cancel()
            try:
                await self.agent_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Agent {self.agent_id} stopped")

    async def receive_message(self, message: Message):
        """Receive a message (called by message bus)"""
        await self.inbox.put(message)

    async def send_message(self, message: Message):
        """Send a message through the message bus"""
        if not self.message_bus:
            raise RuntimeError("Message bus not set")

        message.sender_id = self.agent_id
        await self.message_bus.send_message(message)
        self.metrics['messages_sent'] += 1

    async def send_task_request(self, recipient_id: str, task_type: str,
                                payload: Dict[str, Any], correlation_id: str = None) -> str:
        """Convenience method to send task requests"""
        message = Message(
            type=MessageType.TASK_REQUEST,
            recipient_id=recipient_id,
            task_type=task_type,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        await self.send_message(message)
        return message.correlation_id

    async def send_task_response(self, recipient_id: str, correlation_id: str,
                                 payload: Dict[str, Any], task_type: str = ""):
        """Convenience method to send task responses"""
        message = Message(
            type=MessageType.TASK_RESPONSE,
            recipient_id=recipient_id,
            task_type=task_type,
            payload=payload,
            correlation_id=correlation_id
        )
        await self.send_message(message)

    async def send_error(self, recipient_id: str, correlation_id: str,
                         error: str, details: Dict[str, Any] = None):
        """Convenience method to send error messages"""
        message = Message(
            type=MessageType.TASK_ERROR,
            recipient_id=recipient_id,
            payload={
                'error': error,
                'details': details or {}
            },
            correlation_id=correlation_id
        )
        await self.send_message(message)

    async def _message_loop(self):
        """Main message processing loop for the agent"""
        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self._process_message(message)
                self.metrics['messages_processed'] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} message loop: {e}")
                self.metrics['errors'] += 1
                traceback.print_exc()

    async def _process_message(self, message: Message):
        """Process a received message"""
        try:
            logger.debug(f"Agent {self.agent_id} processing message: {message.id}")

            # Find appropriate handler
            handler = None
            if message.task_type in self.handlers:
                handler = self.handlers[message.task_type]
            elif message.type.value in self.handlers:
                handler = self.handlers[message.type.value]
            else:
                handler = self.handle_message  # Default abstract handler

            # Execute handler
            await handler(message)

        except Exception as e:
            logger.error(f"Error processing message {message.id} in agent {self.agent_id}: {e}")
            # Send error response if this was a task request
            if message.type == MessageType.TASK_REQUEST and message.sender_id:
                await self.send_error(
                    message.sender_id,
                    message.correlation_id or message.id,
                    f"Processing error: {str(e)}"
                )
            self.metrics['errors'] += 1

    async def _handle_system_message(self, message: Message):
        """Handle system messages"""
        logger.info(f"Agent {self.agent_id} received system message: {message.payload}")

    async def _handle_error_message(self, message: Message):
        """Handle error messages"""
        logger.error(f"Agent {self.agent_id} received error: {message.payload}")

    async def _handle_broadcast_message(self, message: Message):
        """Handle broadcast messages"""
        logger.info(f"Agent {self.agent_id} received broadcast: {message.task_type}")
        await self.handle_broadcast(message)

    async def handle_broadcast(self, message: Message):
        """Override to handle broadcast messages"""
        pass

    @abstractmethod
    async def handle_message(self, message: Message):
        """Abstract method - each agent must implement message handling logic"""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        metrics = self.metrics.copy()
        if metrics['start_time']:
            metrics['uptime'] = (datetime.now() - metrics['start_time']).total_seconds()
        return metrics


class SharedContext:
    """Shared context manager for agents that need to share state"""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def set(self, key: str, value: Any):
        """Set a value in shared context"""
        async with self._locks[key]:
            self._data[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context"""
        async with self._locks[key]:
            return self._data.get(key, default)

    async def update(self, key: str, updater: Callable[[Any], Any]):
        """Update a value using a function"""
        async with self._locks[key]:
            current = self._data.get(key)
            self._data[key] = updater(current)

    async def delete(self, key: str):
        """Delete a key from shared context"""
        async with self._locks[key]:
            self._data.pop(key, None)

    async def keys(self) -> List[str]:
        """Get all keys in context"""
        return list(self._data.keys())


# Example utility functions for agent orchestration
class A2AOrchestrator:
    """High-level orchestrator for managing A2A agent workflows"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.shared_context = SharedContext()
        self.agents: Dict[str, BaseA2AAgent] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}

    def register_agent(self, agent: BaseA2AAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)

    def add_workflow(self, workflow_id: str, workflow_config: Dict[str, Any]):
        """Add a workflow configuration"""
        self.workflows[workflow_id] = workflow_config

    async def start_all(self):
        """Start message bus and all agents"""
        await self.message_bus.start()
        for agent in self.agents.values():
            await agent.start()
        logger.info("A2A Orchestrator started all components")

    async def stop_all(self):
        """Stop all agents and message bus"""
        for agent in self.agents.values():
            await agent.stop()
        await self.message_bus.stop()
        logger.info("A2A Orchestrator stopped all components")

    async def execute_workflow(self, workflow_id: str, initial_payload: Dict[str, Any]) -> str:
        """Execute a predefined workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        entry_agent = workflow.get('entry_point')

        if not entry_agent or entry_agent not in self.agents:
            raise ValueError(f"Invalid entry point for workflow {workflow_id}")

        # Create correlation ID for tracking this workflow execution
        correlation_id = str(uuid.uuid4())

        # Send initial message to entry point agent
        message = Message(
            type=MessageType.TASK_REQUEST,
            recipient_id=entry_agent,
            task_type=workflow.get('initial_task_type', 'start_workflow'),
            payload=initial_payload,
            correlation_id=correlation_id
        )

        await self.message_bus.send_message(message)
        return correlation_id
    async def send_message(self, sender_id: str, recipient_id: str, message_type: MessageType,
                          task_type: str, payload: Dict[str, Any], correlation_id: str = None):
        """Send a message through the orchestrator's message bus"""
        message = Message(
            type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            task_type=task_type,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        await self.message_bus.send_message(message)

    async def shutdown(self):
        """Gracefully shutdown the orchestrator (was missing)"""
        await self.stop_all()



# --- Database Management ---
class DatabaseManager:
    def __init__(self, db_path: str = "metadata.db"):
        # FIX: Remove the hardcoded "../metadata.db"
        self.db_path = db_path  # Use the provided path instead
        if not os.path.exists(self.db_path):
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
            -- FIX: Corrected the foreign key reference that was corrupted
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


from enum import Enum
from datetime import timedelta


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for agent communication"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - requests rejected")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.timeout

    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN