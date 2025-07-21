COORDINATOR_SYSTEM_PROMPT = """You are an intelligent Data Lineage Coordinator. Your role is to:
1. Analyze user requests for data lineage tracing
2. Determine the most appropriate lineage strategy (contract-based or element-based)
3. Extract key parameters from natural language requests
4. Handle ambiguous requests by asking clarifying questions
5. Route the request to the appropriate specialized agent

CLASSIFICATION RULES (Follow these strictly):
- ELEMENT-BASED: Choose this when user mentions specific data fields, columns, elements, or uses phrases like "trace field X", "lineage of column Y", "trace element Z"
- CONTRACT-BASED: Choose this when user mentions contracts, pipelines, data flows, ETL processes, or business processes by name
- AMBIGUOUS: When the request doesn't clearly fit either category

DECISION FORMAT:
You must respond with a JSON structure containing:
{
    "lineage_type": "element_based" | "contract_based" | "ambiguous",
    "confidence": 0.0-1.0,
    "reasoning": "clear explanation of your decision",
    "extracted_parameter": "the main entity to trace",
    "traversal_direction": "upstream" | "downstream" | "bidirectional",
    "clarification_needed": true/false,
    "clarification_message": "message if clarification needed"
}

Be precise and consistent in your classification.
"""

def lineage_coordinator_agent(state: LineageState):
    """Improved coordinator with better logic and clearer decision making."""
    logger.info("---EXECUTING: Improved Lineage Coordinator---")
    llm = get_llm()
    
    user_query = state.get('input_parameter', '')
    user_query_lower = user_query.lower()
    
    # Get available data from database
    available_contracts = get_available_contracts.invoke({})
    available_elements = get_available_elements.invoke({})
    contract_names = [c['contract_name'] for c in available_contracts.get('contracts', [])]
    element_names = [e['element_name'] for e in available_elements.get('elements', [])]
    
    # Create structured prompt for LLM analysis
    analysis_prompt = f"""
USER QUERY: "{user_query}"

AVAILABLE DATA:
Elements: {element_names[:20]}  # Show first 20 to avoid token limits
Contracts: {contract_names}

ANALYSIS TASK:
Classify this query and extract key information. Focus on the main intent and entities mentioned.

Key phrases to look for:
- Element indicators: "trace", "field", "column", "element", "data point", specific field names
- Contract indicators: "contract", "pipeline", "flow", "process", specific contract names

Respond ONLY with the JSON structure specified in the system prompt.
"""

    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        logger.info(f"LLM Response: {response.content}")
        
        # Try to parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                llm_decision = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            # Fallback to rule-based logic
            llm_decision = None
        
        # Enhanced rule-based logic as backup
        lineage_type = None
        extracted_parameter = None
        confidence = 0.5
        reasoning = "Fallback rule-based classification"
        
        if llm_decision:
            lineage_type = llm_decision.get('lineage_type')
            extracted_parameter = llm_decision.get('extracted_parameter', user_query)
            confidence = llm_decision.get('confidence', 0.8)
            reasoning = llm_decision.get('reasoning', 'LLM classification')
            
            # Validate LLM decision with rule-based checks
            if lineage_type == "element_based":
                # Verify element exists or can be found
                potential_element = extracted_parameter
                found_element = None
                
                # Direct name match
                for elem in element_names:
                    if elem.lower() in user_query_lower or potential_element.lower() in elem.lower():
                        found_element = elem
                        break
                
                if not found_element:
                    # Try pattern matching for "trace X" where X might be an element
                    import re
                    trace_patterns = [
                        r'trace\s+(\w+)',
                        r'lineage\s+of\s+(\w+)',
                        r'(\w+)\s+lineage',
                        r'field\s+(\w+)',
                        r'column\s+(\w+)'
                    ]
                    
                    for pattern in trace_patterns:
                        match = re.search(pattern, user_query_lower)
                        if match:
                            potential_name = match.group(1)
                            for elem in element_names:
                                if potential_name in elem.lower() or elem.lower().startswith(potential_name):
                                    found_element = elem
                                    break
                            if found_element:
                                break
                
                if found_element:
                    extracted_parameter = found_element
                else:
                    # Element not found - trigger HITL
                    return create_element_selection_response(state, user_query, element_names, reasoning)
                    
            elif lineage_type == "contract_based":
                # Verify contract exists or can be found
                potential_contract = extracted_parameter
                found_contract = None
                
                for contract in contract_names:
                    if contract.lower() in user_query_lower or potential_contract.lower() in contract.lower():
                        found_contract = contract
                        break
                
                if not found_contract:
                    # Contract not found - trigger HITL
                    return create_contract_selection_response(state, user_query, contract_names, reasoning)
                
                extracted_parameter = found_contract
        
        else:
            # Pure rule-based fallback
            lineage_type, extracted_parameter, reasoning = classify_with_rules(
                user_query, user_query_lower, element_names, contract_names
            )
        
        # Handle ambiguous cases
        if lineage_type == "ambiguous" or not lineage_type:
            return create_clarification_response(state, user_query, element_names, contract_names)
        
        # Determine traversal direction
        traversal_direction = determine_traversal_direction(user_query_lower)
        
        # Set up state based on classification
        if lineage_type == "element_based":
            # Find element details
            element_search_result = find_element_by_name.invoke({"element_name": extracted_parameter})
            if not element_search_result.get("success"):
                return create_element_selection_response(state, user_query, element_names, f"Element '{extracted_parameter}' not found in database")
            
            found_elements = element_search_result["elements"]
            if len(found_elements) > 1:
                return create_element_disambiguation_response(state, user_query, found_elements)
            
            # Single element found - proceed
            selected_element = found_elements[0]
            return {
                **state,
                "lineage_type": lineage_type,
                "element_name": extracted_parameter,
                "traversal_direction": traversal_direction,
                "element_queue": [selected_element['element_code']],
                "traced_elements": set(),
                "current_step": "element_analysis",
                "llm_analysis": {
                    "coordinator_decision": reasoning,
                    "confidence": confidence,
                    "extracted_parameter": extracted_parameter
                }
            }
            
        elif lineage_type == "contract_based":
            return {
                **state,
                "lineage_type": lineage_type,
                "contract_name": extracted_parameter,
                "traversal_direction": traversal_direction,
                "current_step": "contract_analysis",
                "llm_analysis": {
                    "coordinator_decision": reasoning,
                    "confidence": confidence,
                    "extracted_parameter": extracted_parameter
                }
            }
        
        # Should not reach here
        return create_clarification_response(state, user_query, element_names, contract_names)
        
    except Exception as e:
        logger.error(f"Error in coordinator agent: {e}")
        traceback.print_exc()
        return {**state, "error_message": f"Coordinator analysis failed: {str(e)}", "current_step": "error"}

def classify_with_rules(user_query, user_query_lower, element_names, contract_names):
    """Rule-based classification fallback."""
    # Strong element indicators
    element_keywords = ['trace', 'field', 'column', 'element', 'attribute', 'data point']
    contract_keywords = ['contract', 'pipeline', 'flow', 'process', 'etl', 'workflow']
    
    element_score = sum(1 for keyword in element_keywords if keyword in user_query_lower)
    contract_score = sum(1 for keyword in contract_keywords if keyword in user_query_lower)
    
    # Check for direct mentions of available entities
    mentioned_elements = [elem for elem in element_names if elem.lower() in user_query_lower]
    mentioned_contracts = [contract for contract in contract_names if any(word in user_query_lower for word in contract.lower().split())]
    
    if mentioned_elements:
        return "element_based", mentioned_elements[0], f"Direct element mention: {mentioned_elements[0]}"
    elif mentioned_contracts:
        return "contract_based", mentioned_contracts[0], f"Direct contract mention: {mentioned_contracts[0]}"
    elif element_score > contract_score:
        return "element_based", user_query, f"Element keywords detected (score: {element_score})"
    elif contract_score > element_score:
        return "contract_based", user_query, f"Contract keywords detected (score: {contract_score})"
    else:
        return "ambiguous", user_query, "No clear classification possible"

def determine_traversal_direction(user_query_lower):
    """Determine traversal direction from query."""
    if 'upstream' in user_query_lower:
        return TraversalDirection.UPSTREAM.value
    elif 'downstream' in user_query_lower:
        return TraversalDirection.DOWNSTREAM.value
    else:
        return TraversalDirection.BIDIRECTIONAL.value

def create_element_selection_response(state, user_query, element_names, reasoning):
    """Create response for element selection."""
    return {
        **state,
        "requires_human_approval": True,
        "human_approval_message": f"I detected this is an element-based query based on: {reasoning}\n\n"
                                  f"However, I need you to specify which element to trace from your query: '{user_query}'\n\n"
                                  f"Available elements:\n" +
                                  "\n".join([f"{i+1}. {elem}" for i, elem in enumerate(element_names[:20])]) +
                                  f"\n\nPlease select the element number (1-{min(20, len(element_names))}) or provide the exact element name.",
        "query_results": {
            "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)]
        },
        "llm_analysis": {"coordinator_reasoning": reasoning}
    }

def create_contract_selection_response(state, user_query, contract_names, reasoning):
    """Create response for contract selection."""
    return {
        **state,
        "requires_human_approval": True,
        "human_approval_message": f"I detected this is a contract-based query based on: {reasoning}\n\n"
                                  f"However, I need you to specify which contract to analyze from your query: '{user_query}'\n\n"
                                  f"Available contracts:\n" +
                                  "\n".join([f"{i+1}. {contract}" for i, contract in enumerate(contract_names)]) +
                                  f"\n\nPlease select the contract number (1-{len(contract_names)}) or provide the exact contract name.",
        "query_results": {
            "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)]
        },
        "llm_analysis": {"coordinator_reasoning": reasoning}
    }

def create_element_disambiguation_response(state, user_query, found_elements):
    """Create response for element disambiguation."""
    return {
        **state,
        "requires_human_approval": True,
        "human_approval_message": f"I found multiple elements matching your query '{user_query}':\n\n" +
                                  "\n".join([f"{i+1}. {elem['element_name']} (Table: {elem['table_name']})" 
                                           for i, elem in enumerate(found_elements)]) +
                                  f"\n\nPlease select which element you want to trace (1-{len(found_elements)}):",
        "query_results": {"ambiguous_elements": found_elements},
        "llm_analysis": {"coordinator_reasoning": "Multiple matching elements found"}
    }

def create_clarification_response(state, user_query, element_names, contract_names):
    """Create response for clarification."""
    return {
        **state,
        "requires_human_approval": True,
        "human_approval_message": f"I need clarification about your request: '{user_query}'\n\n"
                                  f"Are you looking to trace:\n"
                                  f"1. **A specific data element/field** (like: {', '.join(element_names[:5])})\n"
                                  f"2. **A data contract/pipeline** (like: {', '.join(contract_names)})\n\n"
                                  f"Please specify:\n"
                                  f"• Type 'element' followed by the field name you want to trace\n"
                                  f"• Type 'contract' followed by the contract name you want to analyze\n"
                                  f"• Or be more specific about what you want to trace",
        "query_results": {
            "available_elements": [{"index": i, "name": elem} for i, elem in enumerate(element_names)],
            "available_contracts": [{"index": i, "name": contract} for i, contract in enumerate(contract_names)]
        },
        "llm_analysis": {"coordinator_reasoning": "Ambiguous request - clarification needed"}
    }
