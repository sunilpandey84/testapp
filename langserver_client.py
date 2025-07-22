# client.py
import requests
import json
import uuid
import re

# The URL of your LangServe API
API_URL = "http://localhost:8000/lineage/stream"


def run_chat_session():
    """Manages a single chat session with the LangServe API."""
    # Each session needs a unique ID to maintain state
    thread_id = str(uuid.uuid4())
    print(f"Starting new chat session. ID: {thread_id}")
    print("How can I help you trace data lineage today?")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_input = input("> User: ")
            if user_input.lower() == 'quit':
                print("Session ended.")
                break

            # The input to the graph is nested under the 'input' key
            # The session ID is passed in the 'configurable' field
            if "'s" in locals() and "is_resuming" in s and s['is_resuming']:
                # The user is providing feedback to a paused graph
                request_body = {
                    "input": {"human_feedback": {"response": user_input}},
                    "configurable": {"thread_id": thread_id}
                }
            else:
                # This is a new request
                if re.search(r'\b(contract|pipeline)\b', user_input, re.IGNORECASE):
                    lineage_type_val = "contract_based"
                    match = re.search(r"'(.*?)'", user_input)
                    input_param = match.group(1) if match else user_input
                else:
                    lineage_type_val = "element_based"
                    input_param = re.sub(r'trace the element\s*', '', user_input, flags=re.IGNORECASE)

                initial_input = {
                    "lineage_type": lineage_type_val,
                    "input_parameter": input_param,
                    "traversal_direction": "bidirectional",
                    "contract_name": input_param if lineage_type_val == "contract_based" else None,
                    "element_name": input_param if lineage_type_val == "element_based" else None
                }
                request_body = {
                    "input": initial_input,
                    "configurable": {"thread_id": thread_id}
                }

            s = {"is_resuming": False}  # Reset resume flag

            # Make the streaming POST request
            with requests.post(API_URL, stream=True, json=request_body) as response:
                if response.status_code != 200:
                    print(f"Error: Received status code {response.status_code}")
                    print(response.text)
                    continue

                final_output = None
                for line in response.iter_lines():
                    if line:
                        # LangServe streams Server-Sent Events (SSE)
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            try:
                                data_str = decoded_line[len("data:"):].strip()
                                event_data = json.loads(data_str)

                                # The actual graph state is inside the event value
                                state_snapshot = list(event_data.values())[0]
                                final_output = state_snapshot

                                if state_snapshot.get('requires_human_approval'):
                                    print(f"\n🤖 Assistant:\n{state_snapshot['human_approval_message']}")
                                    s['is_resuming'] = True
                                    break  # Break inner loop to prompt user for new input

                            except json.JSONDecodeError:
                                # This can happen with the final "end" event which might not be valid JSON
                                pass

                if not s.get('is_resuming'):
                    # The graph finished without pausing
                    result = final_output.get("final_result", {})
                    print("\n🤖 Assistant (Final Result):")
                    print(json.dumps(result, indent=2))
                    print("\nStarting new trace. How can I help?")

        except requests.ConnectionError:
            print("Connection Error: Is the server running?")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    run_chat_session()