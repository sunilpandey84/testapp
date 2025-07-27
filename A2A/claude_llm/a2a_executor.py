"""
Lineage Analysis Agents - A2A Implementation
Refactored from LangGraph to independent Agent-to-Agent communication
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import your existing database tools and LLM setup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Import our A2A framework (from previous artifact)
from a2a_project.claude_llm.a_2_a_infra import (
    BaseA2AAgent, Message, MessageType, MessagePriority,
    A2AOrchestrator, SharedContext, CircuitBreaker
)

# Import your existing tools and database setup
# (Assuming these are available from your original code)
from a2a_project.claude_llm.a_2_a_infra import (
    query_contract_by_name, query_pipelines_by_contract,
    query_pipeline_dependencies, query_element_mappings_by_queries,
    find_element_by_name, trace_element_connections,
    get_all_query_codes, get_available_contracts, get_available_elements,
    DatabaseManager
)

load_dotenv()
logger = logging.getLogger(__name__)

# Your original system prompts
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
"""


# ... (other system prompts)


class LineageTaskType(Enum):
    """Task types for lineage analysis workflow"""
    ANALYZE_REQUEST = "analyze_request"
    ANALYZE_CONTRACT = "analyze_contract"
    ANALYZE_ELEMENT = "analyze_element"
    CONSTRUCT_LINEAGE = "construct_lineage"
    HANDLE_HUMAN_INPUT = "handle_human_input"
    FINALIZE_RESULTS = "finalize_results"
    HANDLE_ERROR = "handle_error"


@dataclass
class LineageContext:
    """Shared context for lineage analysis workflow"""
    workflow_id: str
    original_query: str
    lineage_type: str = ""
    input_parameter: str = ""
    contract_name: Optional[str] = None
    element_name: Optional[str] = None
    traversal_direction: str = "bidirectional"
    query_results: Dict[str, Any] = None
    lineage_nodes: List[Dict[str, Any]] = None
    lineage_edges: List[Dict[str, Any]] = None
    element_queue: List[str] = None
    traced_elements: Set[str] = None
    requires_human_approval: bool = False
    human_approval_message: Optional[str] = None
    llm_analysis: Dict[str, Any] = None
    recommendations: List[str] = None
    complexity_score: Optional[int] = None
    final_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.query_results is None:
            self.query_results = {}
        if self.lineage_nodes is None:
            self.lineage_nodes = []
        if self.lineage_edges is None:
            self.lineage_edges = []
        if self.element_queue is None:
            self.element_queue = []
        if self.traced_elements is None:
            self.traced_elements = set()
        if self.llm_analysis is None:
            self.llm_analysis = {}
        if self.recommendations is None:
            self.recommendations = []


class LineageCoordinatorAgent(BaseA2AAgent):
    """Intelligent coordinator that analyzes natural language requests using LLM"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="lineage_coordinator",
            name="Lineage Coordinator",
            description="Analyzes user requests and routes to appropriate agents"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()
        self.circuit_breaker = CircuitBreaker()

        # Register task handlers
        self.register_handler(LineageTaskType.ANALYZE_REQUEST.value, self.analyze_request)

    def _get_llm(self):
        """Get LLM instance"""
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.ANALYZE_REQUEST.value:
            await self.analyze_request(message)
        else:
            logger.warning(f"Unknown task type: {message.task_type}")

    async def analyze_request(self, message: Message):
        """Analyze the user request and determine next steps"""
        try:
            logger.info("---EXECUTING: Intelligent Lineage Coordinator---")

            payload = message.payload
            user_query = payload.get('query', '')
            workflow_id = message.correlation_id

            # Create initial context
            context = LineageContext(
                workflow_id=workflow_id,
                original_query=user_query,
                input_parameter=user_query
            )

            # Store context in shared memory
            await self.shared_context.set(workflow_id, context)

            # Get available data from database (using your existing tools)
            available_contracts = await self._get_available_contracts()
            available_elements = await self._get_available_elements()

            contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]
            element_names = [e['element_name'] for e in available_elements.get('elements', [])]

            # Use LLM to analyze the request
            analysis_context = f"""
            User Query: {user_query}
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
                    content=f"{analysis_context}\n\nPlease analyze this request and determine:\n1. Lineage type (contract_based or element_based)\n2. Key parameters to extract\n3. Traversal direction\n4. Any clarifications needed")
            ]

            response = self.llm.invoke(messages)

            # Parse LLM response and make routing decisions
            analysis_result = await self._parse_coordinator_response(
                response.content, user_query, contract_names, element_names, context
            )

            if analysis_result.get('requires_human_input'):
                # Send human input request
                await self.send_task_request(
                    recipient_id="human_approval_agent",
                    task_type=LineageTaskType.HANDLE_HUMAN_INPUT.value,
                    payload={
                        'workflow_id': workflow_id,
                        'message': analysis_result['message'],
                        'options': analysis_result.get('options', {})
                    },
                    correlation_id=workflow_id
                )
            elif analysis_result.get('error'):
                # Send to error handler
                await self.send_task_request(
                    recipient_id="error_handler_agent",
                    task_type=LineageTaskType.HANDLE_ERROR.value,
                    payload={
                        'workflow_id': workflow_id,
                        'error': analysis_result['error']
                    },
                    correlation_id=workflow_id
                )
            else:
                # Route to appropriate analysis agent
                next_agent = analysis_result['next_agent']
                task_type = analysis_result['task_type']

                await self.send_task_request(
                    recipient_id=next_agent,
                    task_type=task_type,
                    payload={
                        'workflow_id': workflow_id,
                        'context': analysis_result.get('context', {})
                    },
                    correlation_id=workflow_id
                )

        except Exception as e:
            logger.error(f"Error in coordinator agent: {e}")
            await self.send_error(
                message.sender_id or "error_handler_agent",
                message.correlation_id,
                f"Coordinator analysis failed: {str(e)}"
            )

    async def _get_available_contracts(self):
        """Get available contracts from database"""
        # This would call your existing tool
        return get_available_contracts.invoke({})

    async def _get_available_elements(self):
        """Get available elements from database"""
        return get_available_elements.invoke({})

    async def _parse_coordinator_response(self, llm_response: str, user_query: str,
                                          contract_names: List[str], element_names: List[str],
                                          context: LineageContext) -> Dict[str, Any]:
        """Parse LLM response and determine routing decisions"""
        content = llm_response.lower()
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

        # Determine traversal direction
        if 'upstream' in content or 'upstream' in user_query_lower:
            direction = "upstream"
        elif 'downstream' in content or 'downstream' in user_query_lower:
            direction = "downstream"
        else:
            direction = "bidirectional"

        context.traversal_direction = direction

        # Determine lineage type and next steps
        if mentioned_element or (
                'trace' in user_query_lower and any(elem.lower() in user_query_lower for elem in element_names)):
            # Element-based lineage
            element_name = mentioned_element
            if not element_name:
                # Try to extract from "trace X lineage" pattern
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
                    'requires_human_input': True,
                    'message': f"I detected this might be an element-based query, but couldn't identify the specific element from your request '{user_query}'.\n\nPlease select from available elements:",
                    'options': {
                        'type': 'element_selection',
                        'available_elements': [{'index': i, 'name': elem} for i, elem in enumerate(element_names)]
                    }
                }

            # Verify element exists in database
            element_search_result = find_element_by_name.invoke({"element_name": element_name})
            if not element_search_result.get("success"):
                return {
                    'requires_human_input': True,
                    'message': f"Could not find element matching '{element_name}' in the database.\n\nAvailable elements are:",
                    'options': {
                        'type': 'element_selection',
                        'available_elements': [{'index': i, 'name': elem} for i, elem in enumerate(element_names)]
                    }
                }

            found_elements = element_search_result["elements"]
            if len(found_elements) > 1:
                return {
                    'requires_human_input': True,
                    'message': f"Found multiple matching elements for '{element_name}'. Please select the correct one:",
                    'options': {
                        'type': 'element_disambiguation',
                        'ambiguous_elements': found_elements
                    }
                }

            # Set up element analysis
            context.lineage_type = "element_based"
            context.element_name = element_name
            context.element_queue = [found_elements[0]['element_code']]
            context.traced_elements = set()

            return {
                'next_agent': 'element_analysis_agent',
                'task_type': LineageTaskType.ANALYZE_ELEMENT.value,
                'context': {'element_code': found_elements[0]['element_code']}
            }

        elif mentioned_contract or any(
                keyword in content or keyword in user_query_lower for keyword in ['contract', 'pipeline']):
            # Contract-based lineage
            contract_name = mentioned_contract if mentioned_contract else user_query

            if not mentioned_contract:
                return {
                    'requires_human_input': True,
                    'message': f"I detected this is a contract-based query, but couldn't identify the specific contract from your request '{user_query}'.\n\nAvailable contracts are:",
                    'options': {
                        'type': 'contract_selection',
                        'available_contracts': [{'index': i, 'name': contract} for i, contract in
                                                enumerate(contract_names)]
                    }
                }

            context.lineage_type = "contract_based"
            context.contract_name = contract_name

            return {
                'next_agent': 'contract_analysis_agent',
                'task_type': LineageTaskType.ANALYZE_CONTRACT.value,
                'context': {'contract_name': contract_name}
            }

        else:
            # Ambiguous request - need clarification
            return {
                'requires_human_input': True,
                'message': f"""I need clarification about your request '{user_query}'.
                Are you looking for:
                1. **Element-based lineage** - Trace a specific data field
                   Available elements: {', '.join(element_names[:10])}{'...' if len(element_names) > 10 else ''}
                2. **Contract-based lineage** - Analyze a data pipeline
                   Available contracts: {', '.join(contract_names)}

                Please specify which type of lineage you need and be more specific about what you want to trace.""",
                'options': {
                    'type': 'lineage_type_selection',
                    'available_elements': [{'index': i, 'name': elem} for i, elem in enumerate(element_names)],
                    'available_contracts': [{'index': i, 'name': contract} for i, contract in enumerate(contract_names)]
                }
            }


class ContractAnalysisAgent(BaseA2AAgent):
    """LLM-powered contract analysis with intelligent insights"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="contract_analysis_agent",
            name="Contract Analysis Agent",
            description="Analyzes data contracts and their pipelines"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()

        # Register task handlers
        self.register_handler(LineageTaskType.ANALYZE_CONTRACT.value, self.analyze_contract)

    def _get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.ANALYZE_CONTRACT.value:
            await self.analyze_contract(message)
        else:
            logger.warning(f"Contract agent received unknown task: {message.task_type}")

    async def analyze_contract(self, message: Message):
        """Analyze contract and gather pipeline data"""
        try:
            logger.info("---EXECUTING: Intelligent Contract Analysis---")

            payload = message.payload
            workflow_id = payload['workflow_id']
            contract_name = payload['context']['contract_name']

            # Get context from shared memory
            context = await self.shared_context.get(workflow_id)
            if not context:
                raise ValueError(f"No context found for workflow {workflow_id}")

            # Gather contract data using existing tools
            contract_details = query_contract_by_name.invoke({"contract_name": contract_name})
            if not contract_details.get("success"):
                await self.send_error(
                    message.sender_id,
                    message.correlation_id,
                    contract_details.get("error", "Contract not found")
                )
                return

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

            response = self.llm.invoke(messages)

            # Update context with results
            context.query_results.update({
                "contract": contract_details,
                "pipelines": pipelines.get("pipelines", []),
                "dependencies": dependencies.get("dependencies", {}),
                "mappings": mappings.get("mappings", [])
            })

            context.llm_analysis["contract_insights"] = response.content
            context.complexity_score = 5  # Could be parsed from LLM response
            context.recommendations = [
                "Consider monitoring data quality in transformation steps",
                "Review pipeline dependencies for optimization opportunities"
            ]

            # Update shared context
            await self.shared_context.set(workflow_id, context)

            # Send to lineage construction
            await self.send_task_request(
                recipient_id="lineage_construction_agent",
                task_type=LineageTaskType.CONSTRUCT_LINEAGE.value,
                payload={'workflow_id': workflow_id},
                correlation_id=workflow_id
            )

        except Exception as e:
            logger.error(f"Error in contract analysis: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Contract analysis failed: {str(e)}"
            )


class ElementAnalysisAgent(BaseA2AAgent):
    """Intelligent element tracing with LLM-powered insights"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="element_analysis_agent",
            name="Element Analysis Agent",
            description="Traces data element connections and transformations"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()

        # Register task handlers
        self.register_handler(LineageTaskType.ANALYZE_ELEMENT.value, self.analyze_element)

    def _get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.ANALYZE_ELEMENT.value:
            await self.analyze_element(message)
        else:
            logger.warning(f"Element agent received unknown task: {message.task_type}")

    async def analyze_element(self, message: Message):
        """Analyze element connections and transformations"""
        try:
            logger.info("---EXECUTING: Intelligent Element Analysis---")

            payload = message.payload
            workflow_id = payload['workflow_id']

            # Get context from shared memory
            context = await self.shared_context.get(workflow_id)
            if not context:
                raise ValueError(f"No context found for workflow {workflow_id}")

            # Process element queue
            if not context.element_queue:
                # No more elements to process, move to construction
                await self.send_task_request(
                    recipient_id="lineage_construction_agent",
                    task_type=LineageTaskType.CONSTRUCT_LINEAGE.value,
                    payload={'workflow_id': workflow_id},
                    correlation_id=workflow_id
                )
                return

            element_to_trace = context.element_queue.pop(0)
            if element_to_trace in context.traced_elements:
                # Continue with next element
                await self.send_task_request(
                    recipient_id="element_analysis_agent",
                    task_type=LineageTaskType.ANALYZE_ELEMENT.value,
                    payload={'workflow_id': workflow_id},
                    correlation_id=workflow_id
                )
                return

            context.traced_elements.add(element_to_trace)

            # Get all available query codes
            all_query_codes_result = get_all_query_codes.invoke({})
            query_codes = all_query_codes_result.get("query_codes", ["Q001", "Q002", "Q003"])

            # Get element mappings with LLM analysis
            all_mappings = query_element_mappings_by_queries.invoke({"query_codes": query_codes})
            related_mappings = [m for m in all_mappings.get("mappings", [])
                                if m['source_code'] == element_to_trace or m['target_code'] == element_to_trace]

            # Use LLM to analyze transformations
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

            response = self.llm.invoke(messages)

            # Update context with new mappings (avoid duplicates)
            if "mappings" not in context.query_results:
                context.query_results["mappings"] = []

            existing_keys = set()
            for em in context.query_results["mappings"]:
                key = (em['query_code'], em['source_code'], em['target_code'])
                existing_keys.add(key)

            # Add only new mappings
            for m in related_mappings:
                key = (m['query_code'], m['source_code'], m['target_code'])
                if key not in existing_keys:
                    context.query_results["mappings"].append(m)
                    existing_keys.add(key)

            # Add LLM insights
            context.llm_analysis[f"element_{element_to_trace}"] = response.content

            # Add newly discovered elements to queue
            for m in related_mappings:
                if m['source_code'] == element_to_trace and m['target_code'] not in context.traced_elements:
                    context.element_queue.append(m['target_code'])
                if m['target_code'] == element_to_trace and m['source_code'] not in context.traced_elements:
                    context.element_queue.append(m['source_code'])

            # Update shared context
            await self.shared_context.set(workflow_id, context)

            # Continue with next element or move to construction
            if context.element_queue:
                await self.send_task_request(
                    recipient_id="element_analysis_agent",
                    task_type=LineageTaskType.ANALYZE_ELEMENT.value,
                    payload={'workflow_id': workflow_id},
                    correlation_id=workflow_id
                )
            else:
                await self.send_task_request(
                    recipient_id="lineage_construction_agent",
                    task_type=LineageTaskType.CONSTRUCT_LINEAGE.value,
                    payload={'workflow_id': workflow_id},
                    correlation_id=workflow_id
                )

        except Exception as e:
            logger.error(f"Error in element analysis: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Element analysis failed: {str(e)}"
            )


class LineageConstructionAgent(BaseA2AAgent):
    """LLM-powered lineage graph construction with optimization"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="lineage_construction_agent",
            name="Lineage Construction Agent",
            description="Builds comprehensive lineage graphs"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()

        # Register task handlers
        self.register_handler(LineageTaskType.CONSTRUCT_LINEAGE.value, self.construct_lineage)

    def _get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        print(f"Task type : {LineageTaskType.CONSTRUCT_LINEAGE.value}")
        print(f"Message Task type : {message.task_type}")
        if message.task_type == LineageTaskType.CONSTRUCT_LINEAGE.value:
            await self.construct_lineage(message)
        else:
            logger.warning(f"Construction agent received unknown task: {message.task_type}")

    async def construct_lineage(self, message: Message):
        """Construct lineage graph from analysis results"""
        try:
            logger.info("---EXECUTING: Intelligent Lineage Construction---")

            payload = message.payload
            workflow_id = payload['workflow_id']

            # Get context from shared memory
            context = await self.shared_context.get(workflow_id)
            if not context:
                raise ValueError(f"No context found for workflow {workflow_id}")

            # Build graph structure
            nodes, edges = [], []
            unique_nodes = {}
            unique_edges = set()

            mappings = context.query_results.get("mappings", [])

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

                # Add unique edges
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

            # Use LLM to analyze graph structure
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
                SystemMessage(content="You are an expert data analyst creating lineage graph analysis."),
                HumanMessage(content=graph_context)
            ]

            response = self.llm.invoke(messages)

            # Update context with constructed graph
            context.lineage_nodes = nodes
            context.lineage_edges = edges
            context.llm_analysis["graph_construction"] = response.content

            # Calculate complexity score
            complexity_score = min((len(edges) + len(nodes)) // 2, 10)
            context.complexity_score = complexity_score

            # Update shared context
            await self.shared_context.set(workflow_id, context)

            # Send to results finalization
            await self.send_task_request(
                recipient_id="finalize_results_agent",
                task_type=LineageTaskType.FINALIZE_RESULTS.value,
                payload={'workflow_id': workflow_id},
                correlation_id=workflow_id
            )

        except Exception as e:
            logger.error(f"Error in lineage construction: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Graph construction failed: {str(e)}"
            )


class HumanApprovalAgent(BaseA2AAgent):
    """Enhanced human interaction with LLM-powered response processing"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="human_approval_agent",
            name="Human Approval Agent",
            description="Handles human-in-the-loop interactions"
        )
        self.shared_context = shared_context
        self.pending_requests: Dict[str, Dict[str, Any]] = {}

        # Register task handlers
        self.register_handler(LineageTaskType.HANDLE_HUMAN_INPUT.value, self.handle_human_input_request)
        self.register_handler("human_feedback", self.process_human_feedback)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.HANDLE_HUMAN_INPUT.value:
            await self.handle_human_input_request(message)
        elif message.task_type == "human_feedback":
            await self.process_human_feedback(message)
        else:
            logger.warning(f"Human approval agent received unknown task: {message.task_type}")

    async def handle_human_input_request(self, message: Message):
        """Handle request for human input"""
        try:
            logger.info("---EXECUTING: Human Approval Agent---")

            payload = message.payload
            workflow_id = payload['workflow_id']
            human_message = payload['message']
            options = payload.get('options', {})

            # Store pending request
            self.pending_requests[workflow_id] = {
                'original_message': message,
                'human_message': human_message,
                'options': options,
                'timestamp': datetime.now()
            }

            # Send human input required response
            await self.send_task_response(
                recipient_id=message.sender_id,
                correlation_id=message.correlation_id,
                payload={
                    'human_input_required': True,
                    'message': human_message,
                    'options': options,
                    'workflow_id': workflow_id
                }
            )

        except Exception as e:
            logger.error(f"Error handling human input request: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Human input processing failed: {str(e)}"
            )

    async def process_human_feedback(self, message: Message):
        """Process human feedback and resume workflow"""
        try:
            payload = message.payload
            workflow_id = payload['workflow_id']
            feedback = payload['feedback']

            if workflow_id not in self.pending_requests:
                await self.send_error(
                    message.sender_id,
                    message.correlation_id,
                    "No pending request found for this workflow"
                )
                return

            pending = self.pending_requests[workflow_id]
            options = pending['options']

            # Get context from shared memory
            context = await self.shared_context.get(workflow_id)
            if not context:
                raise ValueError(f"No context found for workflow {workflow_id}")

            # Process feedback based on option type
            result = await self._process_feedback_by_type(
                options.get('type', ''), feedback, options, context
            )

            if result.get('error'):
                # Send error back to user for correction
                await self.send_task_response(
                    recipient_id=message.sender_id,
                    correlation_id=message.correlation_id,
                    payload={
                        'human_input_required': True,
                        'message': result['error'],
                        'options': options,
                        'workflow_id': workflow_id
                    }
                )
                return

            # Update context with processed result
            if result.get('context_updates'):
                for key, value in result['context_updates'].items():
                    setattr(context, key, value)
                await self.shared_context.set(workflow_id, context)

            # Remove pending request
            del self.pending_requests[workflow_id]

            # Continue workflow to next agent
            next_agent = result['next_agent']
            task_type = result['task_type']

            await self.send_task_request(
                recipient_id=next_agent,
                task_type=task_type,
                payload={
                    'workflow_id': workflow_id,
                    'context': result.get('additional_context', {})
                },
                correlation_id=workflow_id
            )

        except Exception as e:
            logger.error(f"Error processing human feedback: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Feedback processing failed: {str(e)}"
            )

    async def _process_feedback_by_type(self, option_type: str, feedback: Dict[str, Any],
                                        options: Dict[str, Any], context: LineageContext) -> Dict[str, Any]:
        """Process feedback based on the type of selection needed"""

        if option_type == 'element_selection':
            selected_index = feedback.get('selected_index')
            selected_name = feedback.get('selected_name')
            available_elements = options.get('available_elements', [])

            selected_element_name = None
            if selected_index is not None and 0 <= selected_index < len(available_elements):
                selected_element_name = available_elements[selected_index]['name']
            elif selected_name:
                for elem in available_elements:
                    if elem['name'].lower() == selected_name.lower():
                        selected_element_name = elem['name']
                        break

            if not selected_element_name:
                return {'error': 'Invalid selection. Please provide a valid element number or name.'}

            # Search for element in database
            element_search_result = find_element_by_name.invoke({"element_name": selected_element_name})
            if not element_search_result.get("success") or not element_search_result["elements"]:
                return {'error': f'Element "{selected_element_name}" not found in database.'}

            found_elements = element_search_result["elements"]
            return {
                'next_agent': 'element_analysis_agent',
                'task_type': LineageTaskType.ANALYZE_ELEMENT.value,
                'context_updates': {
                    'lineage_type': 'element_based',
                    'element_name': selected_element_name,
                    'element_queue': [found_elements[0]['element_code']],
                    'traced_elements': set()
                }
            }

        elif option_type == 'contract_selection':
            selected_index = feedback.get('selected_index')
            selected_name = feedback.get('selected_name')
            available_contracts = options.get('available_contracts', [])

            selected_contract_name = None
            if selected_index is not None and 0 <= selected_index < len(available_contracts):
                selected_contract_name = available_contracts[selected_index]['name']
            elif selected_name:
                for contract in available_contracts:
                    if contract['name'].lower() == selected_name.lower():
                        selected_contract_name = contract['name']
                        break

            if not selected_contract_name:
                return {'error': 'Invalid selection. Please provide a valid contract number or name.'}

            return {
                'next_agent': 'contract_analysis_agent',
                'task_type': LineageTaskType.ANALYZE_CONTRACT.value,
                'context_updates': {
                    'lineage_type': 'contract_based',
                    'contract_name': selected_contract_name
                },
                'additional_context': {
                    'contract_name': selected_contract_name
                }
            }

        elif option_type == 'element_disambiguation':
            selected_index = feedback.get('selected_index')
            ambiguous_elements = options.get('ambiguous_elements', [])

            if selected_index is None or not (0 <= selected_index < len(ambiguous_elements)):
                return {'error': 'Invalid selection. Please provide a valid element number.'}

            selected_element = ambiguous_elements[selected_index]
            return {
                'next_agent': 'element_analysis_agent',
                'task_type': LineageTaskType.ANALYZE_ELEMENT.value,
                'context_updates': {
                    'lineage_type': 'element_based',
                    'element_name': selected_element['element_name'],
                    'element_queue': [selected_element['element_code']],
                    'traced_elements': set()
                }
            }

        else:
            return {'error': f'Unknown option type: {option_type}'}


# Continuing from where FinalizeResultsAgent was cut off...

class FinalizeResultsAgent(BaseA2AAgent):
    """Final result compilation with LLM-generated summary"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="finalize_results_agent",
            name="Finalize Results Agent",
            description="Compiles final results and generates summaries"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()

        # Register task handlers
        self.register_handler(LineageTaskType.FINALIZE_RESULTS.value, self.finalize_results)

    def _get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.FINALIZE_RESULTS.value:
            await self.finalize_results(message)
        else:
            logger.warning(f"Finalize agent received unknown task: {message.task_type}")

    async def finalize_results(self, message: Message):
        """Compile final results and generate comprehensive summary"""
        try:
            logger.info("---EXECUTING: Finalize Results Agent---")

            payload = message.payload
            workflow_id = payload['workflow_id']

            # Get context from shared memory
            context = await self.shared_context.get(workflow_id)
            if not context:
                raise ValueError(f"No context found for workflow {workflow_id}")

            # Compile comprehensive results
            final_result = {
                "workflow_id": workflow_id,
                "original_query": context.original_query,
                "lineage_type": context.lineage_type,
                "timestamp": datetime.now().isoformat(),
                "lineage_graph": {
                    "nodes": context.lineage_nodes,
                    "edges": context.lineage_edges,
                    "node_count": len(context.lineage_nodes),
                    "edge_count": len(context.lineage_edges)
                },
                "analysis_results": context.query_results,
                "complexity_score": context.complexity_score,
                "recommendations": context.recommendations,
                "llm_insights": context.llm_analysis
            }

            # Generate comprehensive summary using LLM
            summary_context = f"""
            Data Lineage Analysis Complete

            Original Query: {context.original_query}
            Analysis Type: {context.lineage_type}

            Results Summary:
            - Total Nodes: {len(context.lineage_nodes)}
            - Total Edges: {len(context.lineage_edges)}
            - Complexity Score: {context.complexity_score}/10
            - Traversal Direction: {context.traversal_direction}

            Key Findings:
            {json.dumps(context.llm_analysis, indent=2)[:2000]}...

            Graph Structure:
            - Nodes: {[n['name'] for n in context.lineage_nodes[:10]]}{'...' if len(context.lineage_nodes) > 10 else ''}
            - Key Transformations: {[e['transformation'][:50] for e in context.lineage_edges[:5]]}{'...' if len(context.lineage_edges) > 5 else ''}

            Please provide:
            1. Executive summary of the lineage analysis
            2. Key data flow insights
            3. Critical findings and recommendations
            4. Data governance implications
            5. Next steps for optimization
            """

            messages = [
                SystemMessage(content="""You are an expert data analyst providing executive summaries of lineage analysis results. 
                Create comprehensive, actionable insights that help stakeholders understand their data ecosystem."""),
                HumanMessage(content=summary_context)
            ]

            response = self.llm.invoke(messages)

            # Add executive summary to results
            final_result["executive_summary"] = response.content

            # Calculate additional metrics
            final_result["metrics"] = {
                "analysis_duration": (datetime.now() - datetime.fromisoformat(context.workflow_id.split('_')[
                                                                                  1] if '_' in context.workflow_id else datetime.now().isoformat())).total_seconds() if '_' in context.workflow_id else 0,
                "elements_traced": len(context.traced_elements) if context.traced_elements else 0,
                "unique_transformations": len(set(e.get('transformation', '') for e in context.lineage_edges)),
                "data_sources": len(
                    set(n.get('table', '') for n in context.lineage_nodes if n.get('type') == 'source')),
                "data_targets": len(set(n.get('table', '') for n in context.lineage_nodes if n.get('type') == 'target'))
            }

            # Update context with final result
            context.final_result = final_result
            await self.shared_context.set(workflow_id, context)

            # Send completion response
            await self.send_task_response(
                recipient_id=message.sender_id or "client",
                correlation_id=message.correlation_id,
                payload={
                    'status': 'completed',
                    'workflow_id': workflow_id,
                    'results': final_result
                }
            )

            logger.info(f"Lineage analysis completed successfully for workflow: {workflow_id}")

        except Exception as e:
            logger.error(f"Error in finalize results: {e}")
            await self.send_error(
                message.sender_id or "error_handler_agent",
                message.correlation_id,
                f"Results finalization failed: {str(e)}"
            )


class ErrorHandlerAgent(BaseA2AAgent):
    """Enhanced error handling with recovery suggestions and retry logic"""

    def __init__(self, shared_context: SharedContext):
        super().__init__(
            agent_id="error_handler_agent",
            name="Error Handler Agent",
            description="Handles errors and provides recovery suggestions"
        )
        self.shared_context = shared_context
        self.llm = self._get_llm()
        self.error_history: Dict[str, List[Dict[str, Any]]] = {}

        # Register task handlers
        self.register_handler(LineageTaskType.HANDLE_ERROR.value, self.handle_error)
        self.register_handler("retry_workflow", self.retry_workflow)

    def _get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    async def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.task_type == LineageTaskType.HANDLE_ERROR.value:
            await self.handle_error(message)
        elif message.task_type == "retry_workflow":
            await self.retry_workflow(message)
        else:
            logger.warning(f"Error handler received unknown task: {message.task_type}")

    async def handle_error(self, message: Message):
        """Handle errors with intelligent recovery suggestions"""
        try:
            logger.info("---EXECUTING: Error Handler Agent---")

            payload = message.payload
            workflow_id = payload.get('workflow_id', message.correlation_id)
            error_details = payload.get('error', 'Unknown error')
            error_context = payload.get('context', {})

            # Record error in history
            if workflow_id not in self.error_history:
                self.error_history[workflow_id] = []

            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error': error_details,
                'context': error_context,
                'sender': message.sender_id
            }
            self.error_history[workflow_id].append(error_record)

            # Get workflow context if available
            context = await self.shared_context.get(workflow_id)

            # Use LLM to analyze error and suggest recovery
            error_analysis_context = f"""
            Error Analysis for Workflow: {workflow_id}

            Error Details: {error_details}
            Error Context: {json.dumps(error_context, indent=2)}
            Error History: {json.dumps(self.error_history[workflow_id], indent=2)}

            Workflow Context:
            - Original Query: {context.original_query if context else 'Unknown'}
            - Lineage Type: {context.lineage_type if context else 'Unknown'}
            - Current Step: {error_context.get('current_step', 'Unknown')}

            Please analyze this error and provide:
            1. Root cause analysis
            2. Severity assessment (Critical/High/Medium/Low)
            3. Recovery suggestions
            4. Whether retry is recommended
            5. Alternative approaches if retry fails
            """

            messages = [
                SystemMessage(content="""You are an expert error analysis agent for data lineage workflows. 
                Provide actionable recovery suggestions and determine the best path forward."""),
                HumanMessage(content=error_analysis_context)
            ]

            response = self.llm.invoke(messages)

            # Parse LLM response for actionable decisions
            error_analysis = {
                'workflow_id': workflow_id,
                'error_summary': error_details,
                'llm_analysis': response.content,
                'recovery_suggestions': self._extract_recovery_suggestions(response.content),
                'retry_recommended': 'retry' in response.content.lower() and 'not recommended' not in response.content.lower(),
                'severity': self._extract_severity(response.content),
                'timestamp': datetime.now().isoformat()
            }

            # Update context with error information
            if context:
                context.error_message = error_details
                context.llm_analysis['error_analysis'] = error_analysis
                await self.shared_context.set(workflow_id, context)

            # Determine next action based on error analysis
            if error_analysis['retry_recommended'] and len(self.error_history[workflow_id]) < 3:
                # Attempt automatic retry with modified parameters
                await self._attempt_recovery_retry(workflow_id, error_analysis, context)
            else:
                # Send error response to client
                await self.send_task_response(
                    recipient_id=message.sender_id or "client",
                    correlation_id=message.correlation_id,
                    payload={
                        'status': 'error',
                        'workflow_id': workflow_id,
                        'error_analysis': error_analysis,
                        'requires_manual_intervention': True
                    }
                )

        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            # Fallback error response
            await self.send_task_response(
                recipient_id=message.sender_id or "client",
                correlation_id=message.correlation_id,
                payload={
                    'status': 'critical_error',
                    'error': f"Error handler failed: {str(e)}",
                    'requires_immediate_attention': True
                }
            )

    async def _attempt_recovery_retry(self, workflow_id: str, error_analysis: Dict[str, Any],
                                      context: Optional[LineageContext]):
        """Attempt intelligent recovery retry based on error analysis"""
        try:
            logger.info(f"Attempting recovery retry for workflow: {workflow_id}")

            # Reset certain context fields for retry
            if context:
                # Clear error state
                context.error_message = None

                # Reset processing queues if element-based
                if context.lineage_type == "element_based":
                    # Keep traced elements but reset queue
                    context.element_queue = list(context.traced_elements)[-1:] if context.traced_elements else []

                # Update shared context
                await self.shared_context.set(workflow_id, context)

                # Determine retry entry point based on error context
                if context.lineage_type == "element_based":
                    next_agent = "element_analysis_agent"
                    task_type = LineageTaskType.ANALYZE_ELEMENT.value
                elif context.lineage_type == "contract_based":
                    next_agent = "contract_analysis_agent"
                    task_type = LineageTaskType.ANALYZE_CONTRACT.value
                else:
                    next_agent = "lineage_coordinator"
                    task_type = LineageTaskType.ANALYZE_REQUEST.value

                # Send retry request
                await self.send_task_request(
                    recipient_id=next_agent,
                    task_type=task_type,
                    payload={
                        'workflow_id': workflow_id,
                        'retry_attempt': len(self.error_history[workflow_id]),
                        'previous_error': error_analysis
                    },
                    correlation_id=workflow_id
                )

                logger.info(f"Recovery retry initiated for workflow: {workflow_id}")

        except Exception as e:
            logger.error(f"Recovery retry failed: {e}")
            await self.send_task_response(
                recipient_id="client",
                correlation_id=workflow_id,
                payload={
                    'status': 'recovery_failed',
                    'workflow_id': workflow_id,
                    'error': f"Recovery attempt failed: {str(e)}"
                }
            )

    def _extract_recovery_suggestions(self, llm_response: str) -> List[str]:
        """Extract actionable recovery suggestions from LLM response"""
        suggestions = []
        lines = llm_response.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['suggest', 'recommend', 'try', 'consider']):
                if line and not line.startswith('#'):
                    suggestions.append(line)

        return suggestions[:5]  # Limit to top 5 suggestions

    def _extract_severity(self, llm_response: str) -> str:
        """Extract severity level from LLM response"""
        content = llm_response.lower()
        if 'critical' in content:
            return 'Critical'
        elif 'high' in content:
            return 'High'
        elif 'medium' in content:
            return 'Medium'
        else:
            return 'Low'

    async def retry_workflow(self, message: Message):
        """Handle manual retry requests"""
        try:
            payload = message.payload
            workflow_id = payload['workflow_id']
            retry_config = payload.get('retry_config', {})

            logger.info(f"Manual retry requested for workflow: {workflow_id}")

            # Get context and reset error state
            context = await self.shared_context.get(workflow_id)
            if context:
                context.error_message = None
                await self.shared_context.set(workflow_id, context)

                # Start from coordinator with original query
                await self.send_task_request(
                    recipient_id="lineage_coordinator",
                    task_type=LineageTaskType.ANALYZE_REQUEST.value,
                    payload={
                        'query': context.original_query,
                        'manual_retry': True,
                        'retry_config': retry_config
                    },
                    correlation_id=workflow_id
                )
            else:
                raise ValueError(f"No context found for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Manual retry failed: {e}")
            await self.send_error(
                message.sender_id,
                message.correlation_id,
                f"Manual retry failed: {str(e)}"
            )


class LineageA2AOrchestrator:
    """Main orchestrator for the lineage analysis A2A system"""

    def __init__(self):
        self.shared_context = SharedContext()
        self.orchestrator = A2AOrchestrator()
        self.agents = {}
        self._setup_agents()
        self._started = False

    async def start_system(self):
        """Start the message bus and all agents (was missing)"""
        if self._started:
            return
        await self.orchestrator.start_all()
        self._started = True
        logger.info("Lineage A2A system started successfully")

    def _setup_agents(self):
        """Initialize and register all agents"""
        # Create all agents
        self.agents = {
            'lineage_coordinator': LineageCoordinatorAgent(self.shared_context),
            'contract_analysis_agent': ContractAnalysisAgent(self.shared_context),
            'element_analysis_agent': ElementAnalysisAgent(self.shared_context),
            'lineage_construction_agent': LineageConstructionAgent(self.shared_context),
            'human_approval_agent': HumanApprovalAgent(self.shared_context),
            'finalize_results_agent': FinalizeResultsAgent(self.shared_context),
            'error_handler_agent': ErrorHandlerAgent(self.shared_context),
            'client_response_handler': ClientResponseHandler()
        }

        # Register agents with orchestrator
        for agent_id, agent in self.agents.items():
            self.orchestrator.register_agent(agent)

        asyncio.create_task(self.orchestrator.message_bus.add_default_routing())

        logger.info("All lineage analysis agents registered successfully")

    async def start_lineage_analysis(self, user_query: str, user_id: str = "user") -> str:
        """Start a new lineage analysis workflow"""
        try:
            # ENSURE SYSTEM IS STARTED
            if not self._started:
                await self.start_system()
            # Generate unique workflow ID
            workflow_id = f"lineage_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

            logger.info(f"Starting lineage analysis workflow: {workflow_id}")
            logger.info(f"User query: {user_query}")

            # Send initial request to coordinator
            await self.orchestrator.send_message(
                sender_id="client",
                recipient_id="lineage_coordinator",
                message_type=MessageType.TASK_REQUEST,
                task_type=LineageTaskType.ANALYZE_REQUEST.value,
                payload={'query': user_query},
                correlation_id=workflow_id
            )

            return workflow_id

        except Exception as e:
            logger.error(f"Failed to start lineage analysis: {e}")
            raise

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow"""
        try:
            context = await self.shared_context.get(workflow_id)
            if not context:
                return {'status': 'not_found', 'workflow_id': workflow_id}

            status = {
                'workflow_id': workflow_id,
                'original_query': context.original_query,
                'lineage_type': context.lineage_type,
                'status': 'completed' if context.final_result else 'in_progress',
                'complexity_score': context.complexity_score,
                'error_message': context.error_message
            }

            if context.final_result:
                status['results'] = context.final_result

            return status

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {'status': 'error', 'error': str(e)}

    async def handle_human_feedback(self, workflow_id: str, feedback: Dict[str, Any]) -> bool:
        """Handle human feedback for a workflow"""
        try:
            await self.orchestrator.send_message(
                sender_id="client",
                recipient_id="human_approval_agent",
                message_type=MessageType.TASK_REQUEST,
                task_type="human_feedback",
                payload={
                    'workflow_id': workflow_id,
                    'feedback': feedback
                },
                correlation_id=workflow_id
            )
            return True

        except Exception as e:
            logger.error(f"Failed to handle human feedback: {e}")
            return False

    async def retry_workflow(self, workflow_id: str, retry_config: Dict[str, Any] = None) -> bool:
        """Retry a failed workflow"""
        try:
            await self.orchestrator.send_message(
                sender_id="client",
                recipient_id="error_handler_agent",
                message_type=MessageType.TASK_REQUEST,
                task_type="retry_workflow",
                payload={
                    'workflow_id': workflow_id,
                    'retry_config': retry_config or {}
                },
                correlation_id=workflow_id
            )
            return True

        except Exception as e:
            logger.error(f"Failed to retry workflow: {e}")
            return False

    async def wait_for_completion(self, workflow_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Wait for workflow completion with timeout"""
        client_handler = self.agents['client_response_handler']
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for response
            response = client_handler.get_response(workflow_id)
            if response:
                client_handler.clear_response(workflow_id)
                return response

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                return {
                    'status': 'timeout',
                    'error': f'Workflow {workflow_id} timed out after {timeout} seconds'
                }

            # Wait before checking again
            await asyncio.sleep(1.0)

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        if self._started:
            await self.orchestrator.stop_all()
            self._started = False
        logger.info("Lineage A2A Orchestrator shutdown complete")


# Main usage example and client interface
async def main():
    """Example usage of the lineage analysis A2A system"""

    # Initialize the orchestrator
    lineage_orchestrator = LineageA2AOrchestrator()

    try:
        # Start a lineage analysis
        workflow_id = await lineage_orchestrator.start_lineage_analysis(
            "show lineage for order contract",
            user_id="analyst_1"
        )

        print(f"Started workflow: {workflow_id}")

        # Monitor workflow progress
        while True:
            await asyncio.sleep(2)
            status = await lineage_orchestrator.get_workflow_status(workflow_id)
            print(f"Status: {status['status']}")

            if status['status'] == 'completed':
                print("Analysis completed!")
                #print(f"Results: {status.get('results', {}).get('executive_summary', 'No summary available')}")
                print(f"Results: {status.get('results', {})}")
                break
            elif status['status'] == 'error':
                print(f"Analysis failed: {status.get('error_message', 'Unknown error')}")
                break

            # Handle human input if required
            if status.get('human_input_required'):
                # Example of providing human feedback
                feedback = {'selected_index': 0}  # Select first option
                await lineage_orchestrator.handle_human_feedback(workflow_id, feedback)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await lineage_orchestrator.shutdown()


class ClientResponseHandler(BaseA2AAgent):
    """Handles responses back to the client"""

    def __init__(self):
        super().__init__(
            agent_id="client_response_handler",
            name="Client Response Handler",
            description="Handles responses back to the client application"
        )
        self.client_responses: Dict[str, Dict[str, Any]] = {}

    async def handle_message(self, message: Message):
        """Handle responses from other agents"""
        if message.type == MessageType.TASK_RESPONSE:
            workflow_id = message.correlation_id
            self.client_responses[workflow_id] = {
                'status': message.payload.get('status', 'completed'),
                'data': message.payload,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Received client response for workflow: {workflow_id}")
        elif message.type == MessageType.TASK_ERROR:
            workflow_id = message.correlation_id
            self.client_responses[workflow_id] = {
                'status': 'error',
                'error': message.payload.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Received error for workflow: {workflow_id}")

    def get_response(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get response for a workflow"""
        return self.client_responses.get(workflow_id)

    def clear_response(self, workflow_id: str):
        """Clear response for a workflow"""
        self.client_responses.pop(workflow_id, None)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the example
    asyncio.run(main())