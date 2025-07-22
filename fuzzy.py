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

# Add fuzzy matching imports
from difflib import SequenceMatcher
import re
from fuzzywuzzy import fuzz, process  # pip install fuzzywuzzy python-levenshtein

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class FuzzyMatcher:
    """Enhanced fuzzy matching utility for contracts and elements"""
    
    @staticmethod
    def extract_potential_names(query: str) -> List[str]:
        """Extract potential contract/element names from user query"""
        # Remove common words and phrases
        stop_words = {'show', 'me', 'lineage', 'for', 'contract', 'element', 'column', 'track', 'trace', 'the', 'a', 'an'}
        
        # Split by common delimiters and filter
        words = re.split(r'[\s,\.;:]+', query.lower().strip())
        potential_names = [word.strip() for word in words if word.strip() and word.strip() not in stop_words]
        
        # Also try to extract quoted strings or specific patterns
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', query)
        potential_names.extend(quoted_matches)
        
        # Extract after common keywords
        contract_pattern = r'(?:contract|pipeline)\s+([a-zA-Z0-9_]+)'
        element_pattern = r'(?:element|column|field)\s+([a-zA-Z0-9_]+)'
        
        contract_matches = re.findall(contract_pattern, query.lower())
        element_matches = re.findall(element_pattern, query.lower())
        
        potential_names.extend(contract_matches)
        potential_names.extend(element_matches)
        
        return list(set(potential_names))  # Remove duplicates
    
    @staticmethod
    def fuzzy_match_contracts(query: str, available_contracts: List[str], threshold: int = 60) -> List[Dict[str, Any]]:
        """
        Fuzzy match contracts with confidence scores
        Returns list of matches sorted by confidence
        """
        potential_names = FuzzyMatcher.extract_potential_names(query)
        matches = []
        
        for potential_name in potential_names:
            if not potential_name:
                continue
                
            # Use fuzzywuzzy for better matching
            fuzzy_matches = process.extract(potential_name, available_contracts, limit=3)
            
            for match, score in fuzzy_matches:
                if score >= threshold:
                    matches.append({
                        'name': match,
                        'score': score,
                        'query_term': potential_name,
                        'match_type': 'contract'
                    })
        
        # Also try direct substring matching with original query
        query_lower = query.lower()
        for contract in available_contracts:
            contract_lower = contract.lower()
            # Check if contract name appears in query
            if contract_lower in query_lower or any(word in contract_lower for word in query_lower.split()):
                substring_score = fuzz.partial_ratio(query_lower, contract_lower)
                if substring_score >= threshold:
                    matches.append({
                        'name': contract,
                        'score': substring_score,
                        'query_term': query,
                        'match_type': 'contract_substring'
                    })
        
        # Remove duplicates and sort by score
        seen = set()
        unique_matches = []
        for match in matches:
            if match['name'] not in seen:
                unique_matches.append(match)
                seen.add(match['name'])
        
        return sorted(unique_matches, key=lambda x: x['score'], reverse=True)
    
    @staticmethod
    def fuzzy_match_elements(query: str, available_elements: List[str], threshold: int = 60) -> List[Dict[str, Any]]:
        """
        Fuzzy match elements with confidence scores
        Returns list of matches sorted by confidence
        """
        potential_names = FuzzyMatcher.extract_potential_names(query)
        matches = []
        
        for potential_name in potential_names:
            if not potential_name:
                continue
                
            # Use fuzzywuzzy for better matching
            fuzzy_matches = process.extract(potential_name, available_elements, limit=3)
            
            for match, score in fuzzy_matches:
                if score >= threshold:
                    matches.append({
                        'name': match,
                        'score': score,
                        'query_term': potential_name,
                        'match_type': 'element'
                    })
        
        # Also try direct substring matching
        query_lower = query.lower()
        for element in available_elements:
            element_lower = element.lower()
            if element_lower in query_lower or any(word in element_lower for word in query_lower.split()):
                substring_score = fuzz.partial_ratio(query_lower, element_lower)
                if substring_score >= threshold:
                    matches.append({
                        'name': element,
                        'score': substring_score,
                        'query_term': query,
                        'match_type': 'element_substring'
                    })
        
        # Remove duplicates and sort by score
        seen = set()
        unique_matches = []
        for match in matches:
            if match['name'] not in seen:
                unique_matches.append(match)
                seen.add(match['name'])
        
        return sorted(unique_matches, key=lambda x: x['score'], reverse=True)

def lineage_coordinator_agent(state: LineageState):
    """Enhanced intelligent coordinator with fuzzy matching capabilities."""
    logger.info("---EXECUTING: Enhanced Intelligent Lineage Coordinator with Fuzzy Matching---")
    llm = get_llm()
    
    # Get the user query
    user_query = state.get('input_parameter', '')
    
    # Dynamically fetch available data from database
    available_contracts = get_available_contracts.invoke({})
    available_elements = get_available_elements.invoke({})
    contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]
    element_names = [e['element_name'] for e in available_elements.get('elements', [])]
    
    # Enhanced context with fuzzy matching capabilities
    context = f"""
    User Query: {user_query}
    Current State:
    - Messages: {len(state.get('messages', []))} messages in conversation
    - Previous context: {state.get('query_results', {})}
    
    Available Information in Database:
    - Data Contracts: {contract_names}
    - Data Elements: {element_names}
    
    Analysis Rules with Fuzzy Matching:
    - Use fuzzy matching to find best matches for user intent
    - If high-confidence match found (>80), proceed automatically
    - If medium-confidence matches (60-80), ask for confirmation
    - If low-confidence (<60), show options
    """
    
    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{context}\n\nPlease analyze this request and determine:"
                    f"\n1. Lineage type (contract_based or element_based)\n"
                    f"2. Key parameters to extract using fuzzy matching\n"
                    f"3. Traversal direction\n"
                    f"4. Best matching entities from available options")
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
        
        # Use fuzzy matching for contracts
        contract_matches = FuzzyMatcher.fuzzy_match_contracts(user_query, contract_names)
        element_matches = FuzzyMatcher.fuzzy_match_elements(user_query, element_names)
        
        logger.info(f"Contract fuzzy matches: {contract_matches}")
        logger.info(f"Element fuzzy matches: {element_matches}")
        
        # Determine lineage type based on matches and query content
        lineage_type = None
        next_step = None
        
        # Check for high-confidence contract matches
        if contract_matches and contract_matches[0]['score'] >= 80:
            lineage_type = LineageType.CONTRACT_BASED.value
            selected_contract = contract_matches[0]['name']
            
            logger.info(f"High-confidence contract match: {selected_contract} (score: {contract_matches[0]['score']})")
            
            state['contract_name'] = selected_contract
            next_step = "contract_analysis"
        
        # Check for high-confidence element matches  
        elif element_matches and element_matches[0]['score'] >= 80:
            lineage_type = LineageType.ELEMENT_BASED.value
            selected_element = element_matches[0]['name']
            
            logger.info(f"High-confidence element match: {selected_element} (score: {element_matches[0]['score']})")
            
            # Search for the element in database
            element_search_result = find_element_by_name.invoke({"element_name": selected_element})
            if element_search_result.get("success") and element_search_result["elements"]:
                found_elements = element_search_result["elements"]
                state['element_queue'] = [found_elements[0]['element_code']]
                state['traced_elements'] = set()
                state['element_name'] = selected_element
                next_step = "element_analysis"
            else:
                return {
                    **state,
                    "error_message": f"Element '{selected_element}' found by fuzzy matching but not in database",
                    "current_step": "error"
                }
        
        # Medium confidence matches - ask for confirmation
        elif contract_matches and contract_matches[0]['score'] >= 60:
            top_contracts = contract_matches[:3]  # Top 3 matches
            
            confirmation_message = f"""I found these potential contract matches for your query "{user_query}":\n\n"""
            for i, match in enumerate(top_contracts, 1):
                confirmation_message += f"{i}. **{match['name']}** (confidence: {match['score']}%)\n"
            
            confirmation_message += f"\nPlease select the correct contract number (1-{len(top_contracts)}) or type 'none' if none match your intent."
            
            return {
                **state,
                "requires_human_approval": True,
                "human_approval_message": confirmation_message,
                "query_results": {
                    "fuzzy_contract_matches": top_contracts,
                    "match_type": "contract_confirmation"
                },
                "llm_analysis": analysis
            }
        
        elif element_matches and element_matches[0]['score'] >= 60:
            top_elements = element_matches[:3]  # Top 3 matches
            
            confirmation_message = f"""I found these potential element matches for your query "{user_query}":\n\n"""
            for i, match in enumerate(top_elements, 1):
                confirmation_message += f"{i}. **{match['name']}** (confidence: {match['score']}%)\n"
            
            confirmation_message += f"\nPlease select the correct element number (1-{len(top_elements)}) or type 'none' if none match your intent."
            
            return {
                **state,
                "requires_human_approval": True,
                "human_approval_message": confirmation_message,
                "query_results": {
                    "fuzzy_element_matches": top_elements,
                    "match_type": "element_confirmation"
                },
                "llm_analysis": analysis
            }
        
        # No good matches found - show available options
        else:
            # Determine intent from query keywords
            if any(keyword in user_query_lower for keyword in ['contract', 'pipeline']):
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"""I couldn't find a good match for contract-related query: "{user_query}"
                    
Available contracts:
{chr(10).join([f"{i+1}. {contract}" for i, contract in enumerate(contract_names)])}

Please select the contract number or provide a more specific name.""",
                    "query_results": {
                        "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)],
                        "no_fuzzy_matches": True,
                        "attempted_query": user_query
                    },
                    "llm_analysis": analysis
                }
            
            elif any(keyword in user_query_lower for keyword in ['element', 'column', 'field', 'trace']):
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"""I couldn't find a good match for element-related query: "{user_query}"
                    
Available elements:
{chr(10).join([f"{i+1}. {element}" for i, element in enumerate(element_names[:10])])}
{'...' if len(element_names) > 10 else ''}

Please select the element number or provide a more specific name.""",
                    "query_results": {
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)],
                        "no_fuzzy_matches": True,
                        "attempted_query": user_query
                    },
                    "llm_analysis": analysis
                }
            
            else:
                # Ambiguous query - show both options
                return {
                    **state,
                    "requires_human_approval": True,
                    "human_approval_message": f"""I need clarification about your request: "{user_query}"
                    
Are you looking for:
1. **Contract-based lineage** - Analyze a data pipeline/contract
2. **Element-based lineage** - Trace a specific data field/column

Please specify:
- For contracts: {', '.join(contract_names[:5])}{'...' if len(contract_names) > 5 else ''}
- For elements: {', '.join(element_names[:5])}{'...' if len(element_names) > 5 else ''}

Or be more specific about what you want to trace.""",
                    "query_results": {
                        "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)],
                        "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]
                    },
                    "llm_analysis": analysis
                }
        
        # Determine traversal direction using LLM and keywords
        if 'upstream' in content or 'upstream' in user_query_lower:
            direction = TraversalDirection.UPSTREAM.value
        elif 'downstream' in content or 'downstream' in user_query_lower:
            direction = TraversalDirection.DOWNSTREAM.value
        else:
            direction = TraversalDirection.BIDIRECTIONAL.value
        
        logger.info(f"Enhanced coordinator decision: type={lineage_type}, direction={direction}, next_step={next_step}")
        
        return {
            **state,
            "lineage_type": lineage_type,
            "traversal_direction": direction,
            "current_step": next_step,
            "llm_analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced coordinator agent: {e}")
        traceback.print_exc()
        return {**state, "error_message": f"Enhanced coordinator analysis failed: {str(e)}", "current_step": "error"}


def enhanced_human_approval_agent(state: LineageState):
    """Enhanced human interaction with fuzzy matching result processing."""
    logger.info("---EXECUTING: Enhanced Human Approval Agent---")
    
    # If no human approval required, continue
    if not state.get("requires_human_approval"):
        return {**state, "current_step": "error"}
    
    # Check if we have human feedback to process
    human_feedback = state.get("human_feedback")
    if not human_feedback:
        logger.info("Waiting for human feedback...")
        return state
    
    logger.info(f"Processing human feedback: {human_feedback}")
    
    try:
        query_results = state.get("query_results", {})
        
        # Handle fuzzy matching confirmations
        if "fuzzy_contract_matches" in query_results:
            selected_index = human_feedback.get("selected_index")
            if selected_index is not None:
                fuzzy_matches = query_results["fuzzy_contract_matches"]
                if 0 <= selected_index < len(fuzzy_matches):
                    selected_contract = fuzzy_matches[selected_index]["name"]
                    logger.info(f"User confirmed fuzzy match: {selected_contract}")
                    
                    return {
                        **state,
                        "contract_name": selected_contract,
                        "lineage_type": LineageType.CONTRACT_BASED.value,
                        "requires_human_approval": False,
                        "human_feedback": None,
                        "current_step": "contract_analysis"
                    }
        
        elif "fuzzy_element_matches" in query_results:
            selected_index = human_feedback.get("selected_index")
            if selected_index is not None:
                fuzzy_matches = query_results["fuzzy_element_matches"]
                if 0 <= selected_index < len(fuzzy_matches):
                    selected_element_name = fuzzy_matches[selected_index]["name"]
                    logger.info(f"User confirmed fuzzy match: {selected_element_name}")
                    
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
        
        # Handle regular selection cases (fallback to original logic)
        # [Include the rest of the original human_approval_agent logic here...]
        
        # If we reach here, the selection was invalid
        logger.warning("Invalid human feedback received for fuzzy matching")
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": (
                "Invalid selection. Please provide either:\n"
                "• A valid number from the options (e.g., 1, 2, 3)\n"
                "• The exact name of the item you want to select\n\n"
                f"Original message: {state.get('human_approval_message', '')}"
            ),
            "human_feedback": None
        }
        
    except Exception as e:
        logger.error(f"Error processing enhanced human feedback: {e}")
        traceback.print_exc()
        return {
            **state,
            "requires_human_approval": True,
            "human_approval_message": f"Error processing your input: {str(e)}\n\nPlease try again.",
            "human_feedback": None
        }


# Example usage and testing
def test_fuzzy_matching():
    """Test the fuzzy matching functionality"""
    
    # Sample data for testing
    contract_names = ["lars", "calm", "customer_data_contract", "sales_pipeline"]
    element_names = ["col1", "posfactid", "asofdate", "customer_id", "transaction_amount"]
    
    # Test queries
    test_queries = [
        "show me lineage for lars contract",
        "show me lineage for contract calm", 
        "show me lineage for col1 element",
        "show me lineage for posfactid column",
        "track lineage for asofdate element",
        "trace customer_id field upstream",
        "show downstream for lars pipeline"
    ]
    
    print("=== Fuzzy Matching Test Results ===")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        contract_matches = FuzzyMatcher.fuzzy_match_contracts(query, contract_names, threshold=50)
        element_matches = FuzzyMatcher.fuzzy_match_elements(query, element_names, threshold=50)
        
        if contract_matches:
            print(f"  Contract matches: {[(m['name'], m['score']) for m in contract_matches]}")
        if element_matches:
            print(f"  Element matches: {[(m['name'], m['score']) for m in element_matches]}")
        
        if not contract_matches and not element_matches:
            print("  No matches found")

if __name__ == "__main__":
    test_fuzzy_matching()
