from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.database import populate_database_if_empty
from app.graph import compiled_rag_graph, ZeptoAgentResponse

app = FastAPI(
    title="Zepto Operational GenAI Service Engine",
    version="1.0.0",
    description="Automated orchestration layer over store operations policies."
)

class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is the flat delivery fee for orders under INR 149?"])

@app.on_event("startup")
def app_startup_sequence():
    """Trigger the localized document parsing and ingestion loop on startup."""
    populate_database_if_empty()

@app.post("/ask", response_model=ZeptoAgentResponse, summary="Process query through state machine orchestrator.")
def process_agent_query(payload: QueryRequest):
    """Passes queries directly to the compiled LangGraph flow engine."""
    initial_state = {
        "query": payload.query,
        "intent": None,
        "retrieved_context": None,
        "final_output": None,
        "retry_count": 0
    }
    
    try:
        runtime_execution = compiled_rag_graph.invoke(initial_state)
        output_response = runtime_execution.get("final_output")
        
        if not output_response:
            raise HTTPException(status_code=500, detail="Core internal processing pipeline execution state error.")
            
        return output_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline exception error: {str(e)}")
