# Zepto Operational GenAI Service Engine

A lightweight, production-ready, fully self-contained RAG orchestration engine built to handle customer policy verification workflows without relying on external system configurations or active internet access during baseline validation.

---

## Architecture Overview

This service implements a local Retrieval-Augmented Generation (RAG) pipeline with the following stages:

1. **Ingestion** (`app/database.py`)
   - Reads the local policy text files from `docs/`
   - Splits each document by file and stores each file as a single text chunk
   - Builds the local corpus into `ChromaDB` using `chromadb.PersistentClient`

2. **Embedding** (`app/database.py`)
   - Uses `SentenceTransformer('all-MiniLM-L6-v2')`
   - Converts each policy document chunk into a 384-dimensional dense vector
   - Stores document embeddings and metadata in `ChromaDB`

3. **Retrieval** (`app/database.py`)
   - Converts the incoming query into an embedding vector
   - Runs a nearest-neighbor query against the local ChromaDB collection
   - Returns the top 3 matching policy chunks by cosine similarity

4. **Generation** (`app/graph.py`)
   - Uses `LangGraph` to orchestrate the conditional state machine
   - Handles intent classification, retrieval-based answer generation, and fallback direct-answer routes
   - In mock mode, returns a structured canned response without any external LLM call
   - In production mode, calls Groq via `requests.post(...)` and includes a retry-on-validation-failure loop

## MOCK_LLM Branching

The branch is controlled in `app/config.py` via the environment variable:

- `MOCK_LLM=1` (default): offline baseline mode
  - Intent classification is handled by keyword heuristics in `app/graph.py`
  - No external LLM network call is made
  - Responses are deterministic and Pydantic-validated

- `MOCK_LLM=0`: optional real-LLM production mode
  - Uses `execute_groq_completion(...)` in `app/graph.py`
  - Uses the prompt template with the full negative constraint and few-shot example
  - Includes a retry loop for schema validation failures

## Graph Structure

The LangGraph workflow is defined in `app/graph.py` with three named nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

A conditional edge routes the request based on the `intent` value:

- `policy_question` → `retrieve_and_answer`
- `general_question` → `direct_answer`

## Validation Results

The local baseline validation confirmed:

- All 8 `docs/*.txt` files are embedded and queryable in ChromaDB
- `classify_intent(...)` correctly routes a policy-style query and a non-policy query in mock mode
- `retrieve_and_answer(...)` returns actual chunks from the correct source document(s)
- Mock-mode output follows the expected canned templates:
  - `"Based on the retrieved context: ..."`
  - `"I can only answer questions about Zepto policies right now."`
- Pydantic schema is used for the final JSON response (`answer`, `sources`, `confidence`)

## Local Run Instructions

```bash
cd /Users/akinapellikarthik/Desktop/pythonsetup/Capstone_project/support_assistant
source ../.venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Example policy query

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the flat delivery fee for orders under INR 149?"}'
```

### Example out-of-domain query

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Can you give me code to reverse a string in python?"}'
```

## Docker

The repository includes `support_assistant/Dockerfile` for containerized deployment.

Build and run locally:

```bash
cd /Users/akinapellikarthik/Desktop/pythonsetup/Capstone_project/support_assistant
docker build -t zepto-rag-service .
docker run -p 7860:7860 zepto-rag-service
```

The container exposes the `/ask` endpoint on port `7860`.

