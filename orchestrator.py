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
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# FIXED: Enhanced LineageState with better state management
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
    
    # FIXED: Enhanced Human-in-the-Loop (HITL) fields
    requires_human_approval: bool
    human_approval_message: Optional[str]
    human_feedback: Optional[Dict[str, Any]]
    human_approval_type: Optional[str]  # NEW: Track type of approval needed
    pending_action: Optional[str]       # NEW: Track what action to take after approval
    
    # Enhanced fields for LLM responses
    llm_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    complexity_score: Optional[int]
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    
    # NEW: State transition tracking
    previous_step: Optional[str]        # Track previous step for better routing
    next_step_after_approval: Optional[str]  # Explicit next step after human approval


# FIXED: Enhanced human approval agent with proper state transitions
def human_approval_agent(state: LineageState):
    """Enhanced human interaction with proper state transition handling."""
    logger.info("---EXECUTING: Enhanced Human Approval Agent---")
    
    # If no human approval required, continue to next step
    if not state.get("requires_human_approval"):
        # Check if there's a pending next step
        next_step = state.get("next_step_after_approval")
        if next_step:
            logger.info(f"No approval required, proceeding to: {next_step}")
            return {**state, "current_step": next_step}
        return {**state, "current_step": "error"}
    
    # Check if we have human feedback to process
    human_feedback = state.get("human_feedback")
    if not human_feedback:
        # No feedback yet - return state as is (workflow should be paused)
        logger.info("Waiting for human feedback...")
        return state
    
    logger.info(f"Processing human feedback: {human_feedback}")
    
    try:
        query_results = state.get("query_results", {})
        approval_type = state.get("human_approval_type", "unknown")
        
        # FIXED: Handle different types of human approval with proper state transitions
        
        # Handle element disambiguation (multiple matching elements found)
        if approval_type == "element_disambiguation" or "ambiguous_elements" in query_results:
            selected_index = human_feedback.get("selected_index")
            if selected_index is not None:
                elements = query_results.get("ambiguous_elements", [])
                if 0 <= selected_index < len(elements):
                    selected_element = elements[selected_index]
                    logger.info(f"User selected element: {selected_element['element_name']}")
                    
                    # FIXED: Properly set up state for element analysis
                    return {
                        **state,
                        "element_queue": [selected_element["element_code"]],
                        "traced_elements": set(),
                        "element_name": selected_element["element_name"],
                        "lineage_type": "element_based",
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "human_approval_message": None,
                        "human_approval_type": None,
                        "current_step": "element_analysis",
                        "next_step_after_approval": None
                    }
        
        # Handle element selection from available options
        elif approval_type == "element_selection" or "available_elements" in query_results:
            selected_index = human_feedback.get("selected_index")
            selected_name = human_feedback.get("selected_name")
            available_elements = query_results.get("available_elements", [])
            
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
                logger.info(f"User selected element by name: {selected_element_name}")
                
                # Search for the element in database
                element_search_result = find_element_by_name.invoke({"element_name": selected_element_name})
                if element_search_result.get("success") and element_search_result["elements"]:
                    found_elements = element_search_result["elements"]
                    return {
                        **state,
                        "element_queue": [found_elements[0]["element_code"]],
                        "traced_elements": set(),
                        "element_name": selected_element_name,
                        "lineage_type": "element_based",
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "human_approval_message": None,
                        "human_approval_type": None,
                        "current_step": "element_analysis",
                        "next_step_after_approval": None
                    }
        
        # Handle contract selection from available options
        elif approval_type == "contract_selection" or "available_contracts" in query_results:
            selected_index = human_feedback.get("selected_index")
            selected_name = human_feedback.get("selected_name")
            available_contracts = query_results.get("available_contracts", [])
            
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
                logger.info(f"User selected contract: {selected_contract_name}")
                return {
                    **state,
                    "contract_name": selected_contract_name,
                    "lineage_type": "contract_based",
                    "requires_human_approval": False,
                    "human_feedback": None,
                    "human_approval_message": None,
                    "human_approval_type": None,
                    "current_step": "contract_analysis",
                    "next_step_after_approval": None
                }
        
        # FIXED: If we reach here, the selection was invalid - provide better error handling
        logger.warning(f"Invalid human feedback received for approval type: {approval_type}")
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": (
                "Invalid selection. Please provide either:\n"
                "• A valid number from the options (e.g., 1, 2, 3)\n"
                "• The exact name of the item you want to select\n\n"
                f"Original message: {state.get('human_approval_message', '')}"
            ),
            "human_feedback": None,  # Clear invalid feedback
            "current_step": "human_approval"  # Stay in human approval
        }
        
    except Exception as e:
        logger.error(f"Error processing human feedback: {e}")
        traceback.print_exc()
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": f"Error processing your input: {str(e)}\n\nPlease try again.",
            "human_feedback": None,
            "current_step": "human_approval"
        }


# FIXED: Enhanced coordinator agent with proper state management
def lineage_coordinator_agent(state: LineageState):
    """Enhanced intelligent coordinator with proper state transition handling."""
    logger.info("---EXECUTING: Enhanced Intelligent Lineage Coordinator---")
    llm = get_llm()
    
    # Get the user query
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
        
        # FIXED: Determine lineage type with proper state management
        if mentioned_element or ('trace' in user_query_lower and any(elem.lower() in user_query_lower for elem in element_names)):
            lineage_type = "element_based"
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
            
            # FIXED: If no element found, trigger HITL with proper state management
            if not element_name:
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_type": "element_selection",
                    "human_approval_message": f"I detected this might be an element-based query, but couldn't identify the specific element from your request '{user_query}'.\n\n"
                                              f"Please select from available elements:\n" +
                                              "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
                                              f"\n\nOr rephrase your query to be more specific.",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]
                    },
                    "llm_analysis": analysis,
                    "current_step": "human_approval",
                    "next_step_after_approval": "element_analysis"
                }
            
            # Search for the element in database
            element_search_result = find_element_by_name.invoke({"element_name": element_name})
            if not element_search_result.get("success"):
                # FIXED: Trigger HITL for element selection with proper state
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_type": "element_selection",
                    "human_approval_message": f"Could not find element matching '{element_name}' in the database.\n\n"
                                              f"Available elements are:\n" +
                                              "\n".join([f"{i + 1}. {elem}" for i, elem in enumerate(element_names)]) +
                                              f"\n\nPlease select the correct element number or provide a more specific name.",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]
                    },
                    "llm_analysis": analysis,
                    "error_message": f"Element '{element_name}' not found",
                    "current_step": "human_approval",
                    "next_step_after_approval": "element_analysis"
                }
            
            found_elements = element_search_result["elements"]
            if len(found_elements) > 1:
                # FIXED: Enhanced HITL with proper state management
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
                    "human_approval_type": "element_disambiguation",
                    "human_approval_message": clarify_response.content,
                    "query_results": {"ambiguous_elements": found_elements},
                    "llm_analysis": analysis,
                    "current_step": "human_approval",
                    "next_step_after_approval": "element_analysis"
                }
            
            # FIXED: Set up for element tracing with proper state
            next_state = {
                **state,
                "element_queue": [found_elements[0]['element_code']],
                "traced_elements": set(),
                "element_name": element_name,
                "lineage_type": lineage_type,
                "current_step": "element_analysis",
                "llm_analysis": analysis
            }
            
        elif mentioned_contract or any(keyword in content or keyword in user_query_lower for keyword in ['contract', 'pipeline']):
            lineage_type = "contract_based"
            contract_name = mentioned_contract if mentioned_contract else user_query
            
            # FIXED: If no specific contract mentioned, trigger HITL with proper state
            if not mentioned_contract:
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_type": "contract_selection",
                    "human_approval_message": f"I detected this is a contract-based query, but couldn't identify the specific contract from your request '{user_query}'.\n\n"
                                              f"Available contracts are:\n" +
                                              "\n".join([f"{i + 1}. {contract}" for i, contract in enumerate(contract_names)]) +
                                              f"\n\nPlease select the contract number or provide a more specific name.",
                    "query_results": {"available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)]},
                    "llm_analysis": analysis,
                    "current_step": "human_approval",
                    "next_step_after_approval": "contract_analysis"
                }
            
            # FIXED: Set up for contract analysis with proper state
            next_state = {
                **state,
                "contract_name": contract_name,
                "lineage_type": lineage_type,
                "current_step": "contract_analysis",
                "llm_analysis": analysis
            }
            
        else:
            # FIXED: LLM couldn't determine - ask for clarification with proper state
            return {
                **state,
                "requires_human_approval": True,
                "human_approval_type": "lineage_type_selection",
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
                "llm_analysis": analysis,
                "current_step": "human_approval",
                "next_step_after_approval": "coordinator"  # Return to coordinator for re-analysis
            }
        
        # Determine traversal direction
        if 'upstream' in content or 'upstream' in user_query_lower:
            direction = "upstream"
        elif 'downstream' in content or 'downstream' in user_query_lower:
            direction = "downstream"
        else:
            direction = "bidirectional"
        
        next_state["traversal_direction"] = direction
        
        logger.info(f"Coordinator decision: type={lineage_type}, direction={direction}, next_step={next_state['current_step']}")
        
        return next_state
        
    except Exception as e:
        logger.error(f"Error in coordinator agent: {e}")
        traceback.print_exc()
        return {**state, "error_message": f"Coordinator analysis failed: {str(e)}", "current_step": "error"}


# FIXED: Enhanced routing function with better state management
def should_continue(state: LineageState):
    """Enhanced routing with proper state transition handling."""
    current_step = state.get("current_step", "")
    requires_approval = state.get("requires_human_approval", False)
    
    logger.info(f"Routing decision - current_step: {current_step}, requires_approval: {requires_approval}")
    
    # Handle error states
    if current_step == "error":
        return "error_handler"
    
    # Handle human approval requirements
    if requires_approval:
        return "human_approval"
    
    # Handle normal workflow steps
    if current_step == "contract_analysis":
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


# FIXED: Enhanced LineageOrchestrator with better state management
class LineageOrchestrator:
    """Enhanced orchestrator class with improved state management."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.checkpointer = SqliteSaver.from_conn_string("lineage_checkpoints.db")
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.graph = self._build_workflow()
        # FIXED: Enhanced state tracking
        self.paused_state = None
        self.workflow_config = None
        self.current_thread_id = None
    
    def _build_workflow(self):
        """Build the enhanced LangGraph workflow."""
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
        
        # FIXED: Enhanced conditional routing
        workflow.add_conditional_edges("coordinator", should_continue)
        workflow.add_conditional_edges("contract_analysis", should_continue)
        workflow.add_conditional_edges("element_analysis", element_continue_condition)
        workflow.add_conditional_edges("lineage_construction", should_continue)
        workflow.add_conditional_edges("human_approval", should_continue)  # FIXED: Added routing from human_approval
        workflow.add_conditional_edges("finalize_results", should_continue)
        workflow.add_edge("error_handler", END)
        
        # FIXED: Compile with interrupt before human_approval
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_approval"]
        )
    
    async def execute_lineage_request(self, request: LineageRequest) -> Dict[str, Any]:
        """
        Execute a lineage analysis request with enhanced state management.
        """
        # Reset paused state
        self.paused_state = None
        self.workflow_config = None
        self.current_thread_id = None
        
        try:
            # FIXED: Enhanced initial state setup
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
                "human_approval_type": None,
                "pending_action": None,
                "llm_analysis": {},
                "recommendations": [],
                "complexity_score": None,
                "final_result": None,
                "error_message": None,
                "previous_step": None,
                "next_step_after_approval": None
            }
            
            # Create a unique thread ID for this workflow execution
            thread_id = f"thread_{id(request)}_{datetime.now().timestamp()}"
            config = {"configurable": {"thread_id": thread_id}}
            self.workflow_config = config
            self.current_thread_id = thread_id
            
            final_state = None
            
            # FIXED: Enhanced event processing
            async for event in self.graph.astream(initial_state, config):
                logger.info(f"Processing event: {list(event.keys())}")
                
                # Get the state from the event
                for node_name, state_data in event.items():
                    # FIXED: Better tuple handling
                    if isinstance(state_data, tuple):
                        if len(state_data) > 1 and isinstance(state_data[1], dict):
                            state_data = state_data[1]
                        else:
                            logger.warning(f"Unexpected tuple format in state_data: {state_data}")
                            continue
                    
                    # Ensure state_data is a dictionary
                    if not isinstance(state_data, dict):
                        logger.warning(f"state_data is not a dict: {type(state_data)}")
                        continue
                    
                    final_state = state_data
                    
                    # FIXED: Check if human approval is required
                    if state_data.get("requires_human_approval"):
                        logger.info("---WORKFLOW PAUSED FOR HUMAN INPUT---")
                        # Store the current state for resumption
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "message": state_data.get("human_approval_message", "Human input required."),
                            "query_results": state_data.get("query_results", {}),
                            "thread_id": thread_id,
                            "approval_type": state_data.get("human_approval_type"),
                            "current_step": state_data.get("current_step")
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
        """
        Enhanced resume functionality with better state management.
        """
        if not self.paused_state or not self.workflow_config:
            return {"error": "No paused workflow to resume. Please start a new request."}
        
        try:
            logger.info(f"Resuming workflow with feedback: {feedback}")
            
            # FIXED: Better state update for resumption
            updated_state = {**self.paused_state}
            updated_state["human_feedback"] = feedback
            updated_state["previous_step"] = updated_state.get("current_step")
            # Keep requires_human_approval=True so human_approval agent processes it
            updated_state["requires_human_approval"] = True
            
            final_state = None
            
            # FIXED: Resume from current state, not necessarily human_approval
            async for event in self.graph.astream(updated_state, self.workflow_config):
                logger.info(f"Resume event: {list(event.keys())}")
                
                for node_name, state_data in event.items():
                    # Handle tuple format
                    if isinstance(state_data, tuple):
                        if len(state_data) > 1 and isinstance(state_data[1], dict):
                            state_data = state_data[1]
                        else:
                            logger.warning(f"Unexpected tuple format in state_data: {state_data}")
                            continue
                    
                    if not isinstance(state_data, dict):
                        logger.warning(f"state_data is not a dict: {type(state_data)}")
                        continue
                    
                    final_state = state_data
                    
                    # Check if another human approval is needed
                    if (state_data.get("requires_human_approval") and 
                        state_data.get("human_feedback") is None):
                        self.paused_state = state_data
                        return {
                            "human_input_required": True,
                            "message": state_data.get("human_approval_message", "Additional input required."),
                            "query_results": state_data.get("query_results", {}),
                            "approval_type": state_data.get("human_approval_type"),
                            "current_step": state_data.get("current_step")
                        }
            
            # Clear paused state after successful completion
            self.paused_state = None
            self.workflow_config = None
            self.current_thread_id = None
            
            if final_state and final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return {"error": "Workflow resumed but no final result generated"}
                
        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            traceback.print_exc()
            return {"error": f"Failed to resume workflow: {str(e)}"}


# Example usage and testing
async def test_orchestrator():
    """Test the enhanced orchestrator"""
    orchestrator = LineageOrchestrator()
    
    # Test element-based query
    request = LineageRequest(query="show me lineage for col1 element")
    result = await orchestrator.execute_lineage_request(request)
    
    if result.get("human_input_required"):
        print(f"Human input required: {result['message']}")
        # Simulate human feedback
        feedback = {"selected_index": 0}
        final_result = await orchestrator.resume_with_feedback(feedback)
        print(f"Final result: {final_result}")
    else:
        print(f"Direct result: {result}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
