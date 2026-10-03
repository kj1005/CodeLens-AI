# CodeLens AI

### Adaptive Hybrid-RAG Codebase Analysis Assistant

CodeLens AI is an AI-powered codebase analysis tool that allows developers to ask natural-language questions about a GitHub repository. It retrieves relevant source code using hybrid search and uses Gemini to generate grounded answers with file and line-level citations.

## Key Features

- GitHub repository ingestion and code indexing
- Custom line-based code chunking
- Semantic search using BGE embeddings
- BM25 lexical search
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Gemini-powered RAG answers
- Beginner / Intermediate / Expert explanation modes
- Automatic citation verification
- FastAPI backend with Streamlit UI

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embeddings | Sentence Transformers |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Vector Database | ChromaDB |
| Lexical Retrieval | BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Reranker Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Testing | Pytest |
| Version Control | Git / GitHub |

## How It Works

GitHub Repository → Code Chunking + BGE Embeddings → ChromaDB + BM25 → Hybrid Retrieval + RRF → Cross-Encoder Reranking → Gemini RAG Generation → Citation Verification → Answer + Sources

## Adaptive Explanations

CodeLens supports three explanation levels:

- **Beginner** — simple language and explanations of technical terms
- **Intermediate** — implementation flow, functions, and component interactions
- **Expert** — detailed technical implementation and relationships

The retrieved code remains the same while the explanation depth changes according to the selected level.

## Citation Verification

Generated citations are checked against the source-code chunks retrieved by the system.

Supported formats include:

- `[app.py]`
- `[app.py:51]`
- `[app.py:51-72]`

The verifier checks whether the cited file and line range correspond to the retrieved source metadata.

## Setup

### Prerequisites

- Python 3.10+
- Git
- Gemini API key

### 1. Clone the Repository

    git clone https://github.com/kj1005/CodeLens-AI.git
    cd CodeLens-AI

### 2. Create Virtual Environment

    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Configure Gemini

Create `backend/.env` and add:

    GEMINI_API_KEY=your_api_key
    GEMINI_MODEL=gemini-3.8-flash

Do not commit the `.env` file or expose the API key.

## Running the Application

### Start Backend

From the `backend` directory:

    uvicorn main:app --reload --port 8001 --env-file .env

Backend:

    http://127.0.0.1:8001

FastAPI documentation:

    http://127.0.0.1:8001/docs

### Start Streamlit Frontend

Open another terminal in the project root:

    $env:CODELENS_API_URL="http://127.0.0.1:8001"
    backend\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

The Streamlit interface will open at the local URL shown in the terminal, usually:

    http://localhost:8501

## Usage

1. Open the Streamlit application.
2. Enter a public GitHub repository URL.
3. Click **Index Repository**.
4. Ask a natural-language question about the codebase.
5. Select **Beginner**, **Intermediate**, or **Expert** mode.
6. View the answer, retrieved sources, ranking information, and citation verification.

### Example Questions

    Where is authentication handled?
    Where is the database connection initialized?
    How are users created?
    Which function handles authorization?
    Where are the API routes defined?
    How does the application start?

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Backend health check |
| POST | `/repository/index-to-chroma` | Index GitHub repository |
| POST | `/repository/search` | Dense semantic search |
| POST | `/repository/hybrid-search` | Hybrid BM25 + dense search |
| POST | `/repository/ask` | RAG-based question answering |

## Testing

Run the RAG API tests:

    cd backend
    .\.venv\Scripts\python.exe -m pytest tests\test_rag_api.py -q

## Project Structure

    CodeLens-AI/
    │
    ├── backend/
    │   ├── tests/
    │   ├── main.py
    │   ├── rag_api.py
    │   ├── hybrid_api.py
    │   ├── github_service.py
    │   ├── chunking_service.py
    │   ├── embedding_service.py
    │   ├── chroma_service.py
    │   ├── retrieval_service.py
    │   ├── bm25_service.py
    │   ├── reranker_service.py
    │   ├── context_builder.py
    │   ├── gemini_service.py
    │   ├── citation_verification_service.py
    │   ├── requirements.txt
    │   └── .env
    │
    ├── streamlit_app.py
    ├── .gitignore
    └── README.md

## Future Improvements

- AST-aware code chunking
- Function and class-level indexing
- Dependency-aware retrieval
- Incremental repository indexing
- Retrieval and answer-quality evaluation
- Cloud-based vector storage
- Multi-user authentication

## Technical Concepts Demonstrated

RAG · Dense Retrieval · BM25 · Hybrid Search · Reciprocal Rank Fusion · Cross-Encoder Reranking · Vector Databases · Sentence Transformers · LLM Grounding · Citation Verification · FastAPI · Streamlit · REST APIs · GitHub Integration · Automated Testing

## Repository

https://github.com/kj1005/CodeLens-AI

## Author

**Keya Jadhav**  
Computer Engineering Student  
Pimpri Chinchwad College of Engineering, Pune
