# enhanced_main.py - Updated main application with dual chat support

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional, Union
import asyncio
import logging
import uuid
from datetime import datetime
import json

# Import your existing lineage system
from lineageAgentFinal_HTL import LineageOrchestrator, LineageRequest

# Import the new contract creation service
from contreactcreation import (
    ContractCreationService,
    ContractCreationRequest,
    db_manager_global
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Data Assistant API",
    description="Dual-mode API for Data Lineage Analysis and Contract Creation",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200", "http://localhost:3000",
        "http://127.0.0.1:4200", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173"  # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
lineage_orchestrator = LineageOrchestrator()
contract_service = ContractCreationService(db_manager_global)
session_storage = {}  # In production, use Redis or database


# Enhanced session storage with mode support
class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.lineage_message_count = 0
        self.contract_message_count = 0
        self.lineage_orchestrator_state = None
        self.contract_context = {}
        self.current_mode = None


# Pydantic models
class DualModeQueryRequest(BaseModel):
    query: str
    session_id: str
    mode: str  # 'lineage' or 'contract'
    context: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = "ignore"


class LineageQueryRequest(BaseModel):
    query: str
    session_id: str
    context: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = "ignore"


class ContractQueryRequest(BaseModel):
    query: str
    session_id: str
    table_name: Optional[str] = None
    contract_type: Optional[str] = None
    output_format: Optional[str] = "markdown"

    class Config:
        extra = "ignore"


class FeedbackRequest(BaseModel):
    feedback: Dict[str, Any]
    session_id: str
    thread_id: Optional[str] = None

    class Config:
        extra = "ignore"


# Helper functions
def get_or_create_session(session_id: str) -> SessionData:
    """Get or create a session with dual-mode support"""
    if session_id not in session_storage:
        session_storage[session_id] = SessionData(session_id)
    else:
        session_storage[session_id].last_activity = datetime.now()

    return session_storage[session_id]


def update_session_activity(session_id: str, mode: str):
    """Update session activity with mode tracking"""
    if session_id in session_storage:
        session = session_storage[session_id]
        session.last_activity = datetime.now()
        session.current_mode = mode

        if mode == 'lineage':
            session.lineage_message_count += 1
        elif mode == 'contract':
            session.contract_message_count += 1


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Enhanced request logging with mode detection"""
    start_time = datetime.now()

    # Log request details
    logger.info(f"Incoming request: {request.method} {request.url}")

    # Process the request
    response = await call_next(request)

    # Log response time
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Request completed in {process_time:.4f}s with status {response.status_code}")

    return response


# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Smart Data Assistant API is running",
        "version": "2.0.0",
        "modes": ["lineage", "contract"],
        "features": ["dual_chat", "markdown_output", "feedback_support"]
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_storage),
        "version": "2.0.0",
        "services": {
            "lineage_orchestrator": lineage_orchestrator is not None,
            "contract_service": contract_service is not None,
            "database": True  # You can add actual DB health check here
        }
    }


# Dual-mode query endpoint
@app.post("/api/query")
async def process_dual_mode_query(request: DualModeQueryRequest):
    """Process queries in either lineage or contract mode"""
    try:
        logger.info(f"Processing {request.mode} query for session {request.session_id}: {request.query}")

        # Validate mode
        if request.mode not in ['lineage', 'contract']:
            raise HTTPException(
                status_code=400,
                detail="Mode must be either 'lineage' or 'contract'"
            )

        # Get or create session
        session = get_or_create_session(request.session_id)

        if request.mode == 'lineage':
            return await process_lineage_query_internal(request, session)
        else:
            return await process_contract_query_internal(request, session)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing dual-mode query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process query",
                "message": str(e),
                "mode": request.mode,
                "session_id": request.session_id
            }
        )


async def process_lineage_query_internal(request: DualModeQueryRequest, session: SessionData):
    """Internal lineage query processing"""
    # Convert to lineage request format
    context_data = request.context
    if isinstance(context_data, str):
        try:
            context_data = json.loads(context_data)
        except json.JSONDecodeError:
            context_data = {"raw_context": context_data}

    lineage_request = LineageRequest(
        query=request.query,
        context=context_data
    )

    # Process the request
    result = await lineage_orchestrator.execute_lineage_request(lineage_request)

    # Update session activity
    update_session_activity(request.session_id, 'lineage')

    # Store orchestrator state if needed for resumption
    if result.get('human_input_required'):
        session.lineage_orchestrator_state = {
            "paused_state": getattr(lineage_orchestrator, 'paused_state', None),
            "workflow_config": getattr(lineage_orchestrator, 'workflow_config', None)
        }
        if result.get('thread_id'):
            session.lineage_orchestrator_state["thread_id"] = result['thread_id']

    return result


async def process_contract_query_internal(request: DualModeQueryRequest, session: SessionData):
    """Internal contract query processing"""
    # Convert to contract request format
    contract_request = ContractCreationRequest(
        query=request.query,
        table_name=getattr(request, 'table_name', None),
        contract_type=getattr(request, 'contract_type', None),
        output_format='markdown'
    )

    # Process the request
    result = await contract_service.process_contract_request(contract_request)

    # Update session activity
    update_session_activity(request.session_id, 'contract')

    # Store any contract context
    session.contract_context.update({
        "last_query": request.query,
        "result_type": result.get("result_type", "markdown")
    })

    return result


# Legacy lineage endpoints (for backward compatibility)
@app.post("/api/lineage/query")
async def process_lineage_query(request: LineageQueryRequest):
    """Process a lineage query (legacy endpoint)"""
    try:
        logger.info(f"Processing lineage query for session {request.session_id}: {request.query}")

        # Get or create session
        session = get_or_create_session(request.session_id)

        # Create dual-mode request
        dual_request = DualModeQueryRequest(
            query=request.query,
            session_id=request.session_id,
            mode='lineage',
            context=request.context
        )

        return await process_lineage_query_internal(dual_request, session)

    except Exception as e:
        logger.error(f"Error processing lineage query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process lineage query",
                "message": str(e),
                "session_id": request.session_id
            }
        )


@app.post("/api/lineage/feedback")
async def process_lineage_feedback(request: FeedbackRequest):
    """Process human feedback for paused lineage workflow"""
    try:
        logger.info(f"Processing lineage feedback for session {request.session_id}: {request.feedback}")

        # Get session
        if request.session_id not in session_storage:
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please start a new conversation."
            )

        session = session_storage[request.session_id]

        # Restore orchestrator state if available
        if session.lineage_orchestrator_state:
            if hasattr(lineage_orchestrator, 'paused_state'):
                lineage_orchestrator.paused_state = session.lineage_orchestrator_state["paused_state"]
            if hasattr(lineage_orchestrator, 'workflow_config'):
                lineage_orchestrator.workflow_config = session.lineage_orchestrator_state["workflow_config"]

        # Resume with feedback
        result = await lineage_orchestrator.resume_with_feedback(request.feedback)

        # Update session activity
        update_session_activity(request.session_id, 'lineage')

        # Clear orchestrator state if workflow completed
        if not result.get('human_input_required'):
            session.lineage_orchestrator_state = None

        return result

    except Exception as e:
        logger.error(f"Error processing lineage feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process feedback",
                "message": str(e),
                "session_id": request.session_id
            }
        )


# New contract endpoints
@app.post("/api/contract/query")
async def process_contract_query(request: ContractQueryRequest):
    """Process a contract creation query"""
    try:
        logger.info(f"Processing contract query for session {request.session_id}: {request.query}")

        # Get or create session
        session = get_or_create_session(request.session_id)

        # Create contract request
        contract_request = ContractCreationRequest(
            query=request.query,
            table_name=request.table_name,
            contract_type=request.contract_type,
            output_format=request.output_format or 'markdown'
        )

        # Process the request
        result = await contract_service.process_contract_request(contract_request)

        # Update session activity
        update_session_activity(request.session_id, 'contract')

        return result

    except Exception as e:
        logger.error(f"Error processing contract query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process contract query",
                "message": str(e),
                "session_id": request.session_id
            }
        )


@app.get("/api/contract/templates")
async def list_contract_templates():
    """List available contract templates"""
    templates = [
        {
            "id": "generic_template",
            "name": "Generic Data Contract Template",
            "description": "Basic template for any data contract",
            "category": "template"
        },
        {
            "id": "customer_contract",
            "name": "Customer Data Contract",
            "description": "Specialized contract for customer/user data",
            "category": "business_entity"
        },
        {
            "id": "sales_contract",
            "name": "Sales Order Contract",
            "description": "Contract for sales and order data",
            "category": "business_entity"
        },
        {
            "id": "product_contract",
            "name": "Product Catalog Contract",
            "description": "Contract for product and inventory data",
            "category": "business_entity"
        }
    ]

    return {
        "success": True,
        "templates": templates,
        "total_count": len(templates)
    }


@app.post("/api/contract/validate")
async def validate_contract(contract_data: Dict[str, Any]):
    """Validate a contract structure and content"""
    try:
        # Basic validation logic
        required_sections = [
            "contract_name", "version", "owner", "schema_definition"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in contract_data:
                missing_sections.append(section)

        validation_result = {
            "valid": len(missing_sections) == 0,
            "missing_sections": missing_sections,
            "recommendations": [],
            "score": 0
        }

        # Calculate validation score
        score = max(0, 100 - (len(missing_sections) * 25))
        validation_result["score"] = score

        # Add recommendations
        if missing_sections:
            validation_result["recommendations"].append(
                f"Add missing sections: {', '.join(missing_sections)}"
            )

        if score < 75:
            validation_result["recommendations"].append(
                "Consider adding more detailed governance and SLA sections"
            )

        return validation_result

    except Exception as e:
        logger.error(f"Error validating contract: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Contract validation failed: {str(e)}"
        )


# Session management endpoints
@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_storage[session_id]
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "lineage_message_count": session.lineage_message_count,
        "contract_message_count": session.contract_message_count,
        "current_mode": session.current_mode,
        "total_messages": session.lineage_message_count + session.contract_message_count
    }


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
    """List all active sessions with enhanced information"""
    sessions = []
    for session_id, session_data in session_storage.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat(),
            "lineage_messages": session_data.lineage_message_count,
            "contract_messages": session_data.contract_message_count,
            "total_messages": session_data.lineage_message_count + session_data.contract_message_count,
            "current_mode": session_data.current_mode
        })

    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


# Database inspection endpoints (for debugging)
@app.get("/api/database/contracts")
async def list_database_contracts():
    """List contracts available in the database"""
    try:
        conn = db_manager_global.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT v_contract_code, v_contract_name, v_contract_description FROM data_contracts")
        results = cursor.fetchall()
        conn.close()

        contracts = [
            {
                "contract_code": row[0],
                "contract_name": row[1],
                "description": row[2]
            } for row in results
        ]

        return {
            "success": True,
            "contracts": contracts,
            "total_count": len(contracts)
        }

    except Exception as e:
        logger.error(f"Error listing database contracts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve contracts: {str(e)}"
        )


@app.get("/api/database/elements")
async def list_database_elements():
    """List data elements available in the database"""
    try:
        conn = db_manager_global.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT v_data_element_name, v_data_element_code, v_table_name
                       FROM business_element_mapping
                       ORDER BY v_data_element_name LIMIT 100
                       """)
        results = cursor.fetchall()
        conn.close()

        elements = [
            {
                "element_name": row[0],
                "element_code": row[1],
                "table_name": row[2]
            } for row in results
        ]

        return {
            "success": True,
            "elements": elements,
            "total_count": len(elements)
        }

    except Exception as e:
        logger.error(f"Error listing database elements: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve elements: {str(e)}"
        )


# Analytics and metrics endpoints
@app.get("/api/analytics/usage")
async def get_usage_analytics():
    """Get usage analytics across both modes"""
    total_sessions = len(session_storage)
    lineage_usage = sum(1 for s in session_storage.values() if s.current_mode == 'lineage')
    contract_usage = sum(1 for s in session_storage.values() if s.current_mode == 'contract')

    total_lineage_messages = sum(s.lineage_message_count for s in session_storage.values())
    total_contract_messages = sum(s.contract_message_count for s in session_storage.values())

    return {
        "total_sessions": total_sessions,
        "active_modes": {
            "lineage": lineage_usage,
            "contract": contract_usage
        },
        "message_counts": {
            "lineage_messages": total_lineage_messages,
            "contract_messages": total_contract_messages,
            "total_messages": total_lineage_messages + total_contract_messages
        },
        "most_popular_mode": "lineage" if lineage_usage >= contract_usage else "contract",
        "timestamp": datetime.now().isoformat()
    }


# Test endpoints for debugging
@app.post("/api/test/echo")
async def echo_request(request: Request):
    """Echo back the request for debugging purposes"""
    try:
        body = await request.body()
        headers = dict(request.headers)

        return {
            "method": request.method,
            "url": str(request.url),
            "headers": headers,
            "body_raw": body.decode() if body else None,
            "body_json": json.loads(body) if body else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "method": request.method,
            "url": str(request.url),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/test/contract")
async def test_contract_creation():
    """Test contract creation functionality"""
    try:
        test_request = ContractCreationRequest(
            query="Create a test contract for user table",
            table_name="users",
            contract_type="test",
            output_format="markdown"
        )

        result = await contract_service.process_contract_request(test_request)
        return {
            "test_status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Enhanced error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "message": "The request data format is invalid",
            "details": exc.errors(),
            "url": str(request.url),
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "url": str(request.url),
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Smart Data Assistant API starting up...")
    logger.info(f"Lineage orchestrator initialized: {lineage_orchestrator is not None}")
    logger.info(f"Contract service initialized: {contract_service is not None}")
    logger.info("Dual-mode chat interface ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Smart Data Assistant API shutting down...")
    # Clean up any resources if needed
    session_storage.clear()


# Main execution
if __name__ == "__main__":
    import uvicorn

    # Run the server with enhanced configuration
    uvicorn.run(
        "enhanced_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )