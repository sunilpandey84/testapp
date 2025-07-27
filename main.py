# main.py - FastAPI Backend for React Chatbot
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
import uuid
from datetime import datetime

# Import your existing lineage system
from lineageAgentFinal_HTL import LineageOrchestrator, LineageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Lineage Chat API",
    description="API for Data Lineage Chat Assistant",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000",
                   "http://127.0.0.1:4200", "http://127.0.0.1:3000"],  # React app URLs3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance and session storage
orchestrator = LineageOrchestrator()
session_storage = {}  # In production, use Redis or database


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: str
    context: Optional[str] = None


class FeedbackRequest(BaseModel):
    feedback: Dict[str, Any]
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int


# Helper functions
def get_or_create_session(session_id: str) -> Dict:
    """Get or create a session"""
    if session_id not in session_storage:
        session_storage[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
            "orchestrator_state": None
        }
    else:
        session_storage[session_id]["last_activity"] = datetime.now()

    return session_storage[session_id]


def update_session_activity(session_id: str):
    """Update session activity timestamp"""
    if session_id in session_storage:
        session_storage[session_id]["last_activity"] = datetime.now()
        session_storage[session_id]["message_count"] += 1


# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Lineage Chat API is running", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_storage),
        "version": "1.0.0"
    }


@app.post("/api/lineage/query")
async def process_lineage_query(request: QueryRequest):
    """Process a lineage query"""
    try:
        logger.info(f"Processing query for session {request.session_id}: {request.query}")

        # Get or create session
        session = get_or_create_session(request.session_id)

        # Create lineage request
        lineage_request = LineageRequest(
            query=request.query,
            context=request.context
        )

        # Process the request
        result = await orchestrator.execute_lineage_request(lineage_request)

        # Update session activity
        update_session_activity(request.session_id)

        # Store orchestrator state if needed for resumption
        if result.get('human_input_required'):
            session["orchestrator_state"] = {
                "paused_state": orchestrator.paused_state,
                "workflow_config": orchestrator.workflow_config
            }

        logger.info(f"Query processed successfully for session {request.session_id}")
        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process lineage query",
                "message": str(e),
                "session_id": request.session_id
            }
        )


@app.post("/api/lineage/feedback")
async def process_feedback(request: FeedbackRequest):
    """Process human feedback for paused workflow"""
    try:
        logger.info(f"Processing feedback for session {request.session_id}: {request.feedback}")

        # Get session
        if request.session_id not in session_storage:
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please start a new conversation."
            )

        session = session_storage[request.session_id]

        # Restore orchestrator state if available
        if session.get("orchestrator_state"):
            orchestrator.paused_state = session["orchestrator_state"]["paused_state"]
            orchestrator.workflow_config = session["orchestrator_state"]["workflow_config"]

        # Resume with feedback
        result = await orchestrator.resume_with_feedback(request.feedback)

        # Update session activity
        update_session_activity(request.session_id)

        # Clear orchestrator state if workflow completed
        if not result.get('human_input_required'):
            session["orchestrator_state"] = None
        else:
            # Update state for potential further feedback
            session["orchestrator_state"] = {
                "paused_state": orchestrator.paused_state,
                "workflow_config": orchestrator.workflow_config
            }

        logger.info(f"Feedback processed successfully for session {request.session_id}")
        return result

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process feedback",
                "message": str(e),
                "session_id": request.session_id
            }
        )


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_storage[session_id]
    return SessionInfo(
        session_id=session["session_id"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        message_count=session["message_count"]
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in session_storage:
        del session_storage[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session_data in session_storage.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_activity": session_data["last_activity"].isoformat(),
            "message_count": session_data["message_count"]
        })

    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.post("/api/sessions/cleanup")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """Cleanup sessions older than specified hours"""
    current_time = datetime.now()
    sessions_to_remove = []

    for session_id, session_data in session_storage.items():
        age_hours = (current_time - session_data["last_activity"]).total_seconds() / 3600
        if age_hours > max_age_hours:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        del session_storage[session_id]

    return {
        "message": f"Cleaned up {len(sessions_to_remove)} old sessions",
        "cleaned_sessions": sessions_to_remove,
        "remaining_sessions": len(session_storage)
    }


# Database inspection endpoints (for debugging and admin)
@app.get("/api/database/contracts")
async def get_available_contracts():
    """Get all available contracts from database"""
    try:
        from lineageAgentFinal_HTL import get_available_contracts
        result = get_available_contracts.invoke({})
        return result
    except Exception as e:
        logger.error(f"Error fetching contracts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/elements")
async def get_available_elements():
    """Get all available data elements from database"""
    try:
        from lineageAgentFinal_HTL import get_available_elements
        result = get_available_elements.invoke({})
        return result
    except Exception as e:
        logger.error(f"Error fetching elements: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/queries")
async def get_available_queries():
    """Get all available query codes from database"""
    try:
        from lineageAgentFinal_HTL import get_all_query_codes
        result = get_all_query_codes.invoke({})
        return result
    except Exception as e:
        logger.error(f"Error fetching query codes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Data Lineage Chat API starting up...")
    logger.info(f"Orchestrator initialized: {orchestrator is not None}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Data Lineage Chat API shutting down...")
    # Clean up any resources if needed
    session_storage.clear()


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )