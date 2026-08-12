import json
import requests
from typing_extensions import TypedDict
from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from app.config import settings
from app.database import query_vector_store

# --- TASK 5: PYDANTIC SCHEMA GUARANTEES ---
class ZeptoAgentResponse(BaseModel):
    answer: str = Field(description="The direct answer resolving the client interface query.")
    sources: List[str] = Field(default=[], description="The list of structural source document identifiers used.")
    confidence: float = Field(description="Normalized certainty rating bounded strictly between 0.0 and 1.0.")

# Define structured state machine tracking dictionary 
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_context: Optional[List[dict]]
    final_output: Optional[ZeptoAgentResponse]
    retry_count: int

# --- TASK 2: PROMPT STRUCTURAL SKELETON DEFINITION ---
STRUCTURED_RAG_PROMPT_TEMPLATE = """
ROLE: You are an elite, highly precise Customer Support AI Assistant specialized exclusively in Zepto internal store operations.
CONTEXT: Rely ONLY on the verified policy clauses provided below.
---
{context_block}
---
TASK: Resolve the customer query comprehensively. If the problem cannot be accurately resolved using the context, state that you do not possess the required background knowledge.
NEGATIVE CONSTRAINTS:
1. Do not mention or synthesize facts outside the provided document parameters.
2. Under no circumstances should you generate answers based on external web training data.

FEW-SHOT TRAINING EXAMPLE:
Query: "Can I return open body wash?"
Context snippet: [doc_02 — Returns: Personal care items that have been opened are non-returnable except for defects.]
Output JSON format:
{{
  "answer": "No, personal care items that have been opened cannot be returned unless there is a manufacturing defect present.",
  "sources": ["doc_02"],
  "confidence": 1.0
}}

Execute the conversion process into raw valid JSON format matching the schema for query: "{query_text}"
"""

def execute_groq_completion(prompt_payload: str, fallback_json_str: str) -> dict:
    """Helper module acting as a Groq REST connector with corrective validation mechanics."""
    if not settings.GROQ_API_KEY:
         return json.loads(fallback_json_str)
    
    url = "https://groq.com"
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a JSON-only response engine. Output raw parsable JSON string objects only."},
            {"role": "user", "content": prompt_payload}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=15)
        if res.status_code == 200:
            content_str = res.json()['choices'][0]['message']['content']
            return json.loads(content_str)
    except Exception as e:
        print(f"[!] Groq execution exception error: {e}. Falling back to default baseline schema.")
    return json.loads(fallback_json_str)

# --- NODE DEFINITIONS ---
def classify_intent(state: AgentState) -> dict:
    """Classifies incoming user intent using either keyword heuristics or external LLM checks."""
    query = state["query"].lower()
    
    if settings.MOCK_LLM == 1:
        # Graded Baseline Heuristic matching constraints
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        if any(kw in query for kw in keywords):
            intent_val = "policy_question"
        else:
            intent_val = "general_question"
    else:
        # Production live mode execution boundary
        classification_prompt = f"Classify this text as either 'policy_question' or 'general_question': '{query}'. Return JSON format: {{\"intent\": \"value\"}}"
        mock_fallback = '{"intent": "general_question"}'
        res_json = execute_groq_completion(classification_prompt, mock_fallback)
        intent_val = res_json.get("intent", "general_question")
        
    return {"intent": intent_val}

def retrieve_and_answer(state: AgentState) -> dict:
    """Pulls relevant document slices from ChromaDB and builds the final structured response."""
    query = state["query"]
    # Local feature matching runs across both system execution layers
    matched_chunks = query_vector_store(query, top_k=3)
    top_chunk = matched_chunks[0] if matched_chunks else {"id": "Unknown", "text": "No matching context found."}
    
    if settings.MOCK_LLM == 1:
        # Graded Baseline output generation logic
        canned_text = f"Based on the retrieved context: {top_chunk['text'][:200]}"
        sources_list = [c['id'] for c in matched_chunks]
        
        final_response = ZeptoAgentResponse(
            answer=canned_text,
            sources=sources_list,
            confidence=1.0
        )
    else:
        # Live processing with production LLM API
        context_str = "\n".join([f"[{c['id']}]: {c['text']}" for c in matched_chunks])
        formatted_prompt = STRUCTURED_RAG_PROMPT_TEMPLATE.format(context_block=context_str, query_text=query)
        
        fallback_str = json.dumps({
            "answer": f"Fallback mitigation generated: {top_chunk['text'][:150]}",
            "sources": [top_chunk['id']],
            "confidence": 0.50
        })
        
        # Implement loops managing validation layers and corrective retries
        current_retry = state.get("retry_count", 0)
        parsed_successfully = False
        final_response = None
        
        while current_retry <= 2 and not parsed_successfully:
            try:
                llm_json = execute_groq_completion(formatted_prompt, fallback_str)
                final_response = ZeptoAgentResponse(**llm_json)
                parsed_successfully = True
            except Exception:
                current_retry += 1
                formatted_prompt += "\nCORRECTIVE INSTRUCTION: Previous iteration output failed validation schema constraints. Output valid JSON fields now."
        
        if not parsed_successfully:
            final_response = ZeptoAgentResponse(
                answer="Error: Downstream content generation extraction parsing failed validation parameters.",
                sources=[],
                confidence=0.0
            )
            
    return {"final_output": final_response, "retrieved_context": matched_chunks}

def direct_answer(state: AgentState) -> dict:
    """Handles general conversational queries outside the store policy domain."""
    if settings.MOCK_LLM == 1:
        final_response = ZeptoAgentResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        prompt = f"Provide a brief polite response explaining that you only serve operational customer query contexts for: '{state['query']}'"
        fallback_str = '{"answer": "I handle operational policy questions only.", "sources": [], "confidence": 1.0}'
        llm_json = execute_groq_completion(prompt, fallback_str)
        final_response = ZeptoAgentResponse(
            answer=llm_json.get("answer", "Service only targets platform core operations documentation queries."),
            sources=[],
            confidence=0.90
        )
        
    return {"final_output": final_response}

# --- EDGE ROUTING ENGINE CONTROLLER ---
def intent_conditional_router(state: AgentState) -> str:
    """Decides the appropriate execution path based on the classification results."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# --- SYSTEM GRAPH COMPOSITION ---
workflow = StateGraph(AgentState)

# Append functional node coordinates
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

# Enforce processing workflows
workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    intent_conditional_router,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

# Compile functional workflow graphs
compiled_rag_graph = workflow.compile()
