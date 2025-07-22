# server.py
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
import os

# Import the orchestrator from your agentic code
from final.lineageAgentFinal_HTL import LineageOrchestrator, LineageType, TraversalDirection

# --- FastAPI App Setup ---
app = FastAPI(
    title="Data Lineage Agent API",
    version="1.0",
    description="An API for an agentic data lineage tracing system.",
)

# --- Orchestrator and Graph Instantiation ---
# Ensure the API key is set before initializing
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it and restart the app.")
orchestrator = LineageOrchestrator()
lineage_graph = orchestrator.graph

# --- LangServe Route ---
# This exposes the compiled graph at the '/lineage' endpoint.
# The `config_keys` argument is crucial for enabling stateful, multi-turn conversations.
add_routes(
    app,
    lineage_graph,
    path="/lineage",
    config_keys=["configurable"], # This allows passing session IDs
)

# --- Main Execution ---
if __name__ == "__main__":
    # This runs the API server
    uvicorn.run(app, host="localhost", port=8000)