from flask import Flask, request, jsonify
from flask_cors import CORS
from conversational_lineage_agent import ConversationalLineageAgent
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the agent with web interface mode
agent = ConversationalLineageAgent(interface_mode="web")

@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint for processing user queries"""
    try:
        logger.info(f"Received request headers: {dict(request.headers)}")
        logger.info(f"Received request data: {request.data}")
        
        data = request.json
        if not data or 'message' not in data:
            logger.warning("No message provided in request")
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        user_message = data['message']
        feedback = data.get('feedback', None)  # Get feedback if provided
        metadata_selection = data.get('metadata_selection', None)  # Get metadata selection flag
        original_query = data.get('original_query', None)  # Get original query for feedback loop
        
        logger.info(f"Processing query: {user_message}")
        
        # If this is a feedback response to an uncertain answer
        if feedback and original_query:
            logger.info(f"Processing feedback: {feedback} for query: {original_query}")
            # Get the agent's internal instance for handling feedback
            response = agent.process_feedback(original_query, feedback)
        # If this is a metadata selection response
        elif metadata_selection and original_query:
            logger.info(f"Processing metadata selection: {metadata_selection} for query: {original_query}")
            # Process as feedback, but mark it as a metadata selection
            # We'll use the same feedback mechanism but with a special prefix to indicate it's a metadata selection
            selection_feedback = f"I select: {user_message}"
            logger.info(f"Sending selection feedback: '{selection_feedback}' for original query: '{original_query}'")
            
            # Log the current conversation state before processing
            logger.info(f"Current agent conversation state before processing: {len(agent.conversation_state['messages'])} messages")
            
            response = agent.process_feedback(original_query, selection_feedback)
            
            # Log response type to help diagnose issues
            logger.info(f"Metadata selection response type: {type(response)}")
            logger.info(f"Metadata selection response preview: {response[:100]}...")
        else:
            # Process the message using the agent
            response = agent.process_message(user_message)
            
            # Add debug logging
            logger.info(f"Agent response type: {type(response)}")
            logger.info(f"Agent response: {response[:200]}...")  # Log first 200 chars to avoid flooding logs
        
        # Check if response is a JSON string (uncertainty or metadata selection response)
        try:
            import json
            is_json_response = isinstance(response, str) and response.strip().startswith('{')
            logger.info(f"Is JSON response: {is_json_response}")
            
            response_obj = json.loads(response) if is_json_response else None
            if response_obj:
                logger.info(f"JSON keys in response: {list(response_obj.keys())}")
                logger.info(f"needs_metadata_selection: {response_obj.get('needs_metadata_selection', False)}")
            
            # Handle uncertainty response
            if response_obj and isinstance(response_obj, dict) and response_obj.get('needs_clarification', False):
                logger.info("Detected uncertainty in response, requesting clarification from UI")
                return jsonify({
                    'success': True,
                    'needs_clarification': True,
                    'response': response_obj.get('original_response', ''),
                    'original_query': response_obj.get('original_query', ''),
                    'reasoning': response_obj.get('reasoning', 'Additional information is needed to provide a complete answer.')
                })
                
            # Handle metadata selection response
            if response_obj and isinstance(response_obj, dict) and response_obj.get('needs_metadata_selection', False):
                logger.info("Detected need for metadata selection, providing options to user")
                return jsonify({
                    'success': True,
                    'needs_metadata_selection': True,
                    'original_query': response_obj.get('original_query', ''),
                    'formatted_options': response_obj.get('formatted_options', ''),
                    'options': response_obj.get('options', {}),
                    'reasoning': response_obj.get('reasoning', 'Please select from the available metadata options.')
                })
        except (json.JSONDecodeError, TypeError):
            # Not a JSON special response, continue normally
            pass
            
        logger.info(f"Generated response length: {len(response) if response else 0}")
        
        result = {
            'success': True,
            'response': response
        }
        logger.info("Returning successful response")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    logger.info("Starting Lineage Agent API server...")
    app.run(debug=True, host='0.0.0.0', port=5002)
