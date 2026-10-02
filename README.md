# CodeLens AI

CodeLens AI is a full-stack GitHub codebase analysis assistant. The backend ingests supported repository files, chunks their source, generates local code embeddings, persists vectors in ChromaDB, and retrieves relevant chunks through dense vector search. Answer generation is not implemented.

## Embeddings

An embedding is a numerical representation of a text passage. Here, each code chunk is encoded from its source text so a later step can store and compare those vectors. The model is `BAAI/bge-small-en-v1.5`; it produces 384-dimensional vectors and runs on CPU. The model is loaded once per backend process and downloaded from Hugging Face the first time an embedding request is made.

## Backend setup

From the project root, create the environment and install the backend requirements:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API from the `backend` directory:

```powershell
uvicorn main:app --reload
```

The health check is available at <http://127.0.0.1:8000/>. The API documentation is at <http://127.0.0.1:8000/docs>.

## Test repository embeddings

In another PowerShell terminal, submit a public GitHub repository URL:

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/repository/embeddings `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"url":"https://github.com/owner/repository","max_chunks":5}' |
	ConvertTo-Json -Depth 8
```

The response includes the repository name, supported file and chunk totals, embedding dimension, and up to the requested number of chunks with their metadata and vectors. `max_chunks` defaults to 100 and can be set up to 500. The first embedding request may take longer while the model weights are downloaded.

## ChromaDB vector storage

ChromaDB persists the vectors generated from code chunks in `backend/chroma_data/`. Each record uses the chunk ID as its ID, the original source code as its document, the 384-dimensional embedding as its vector, and file path, language, line range, and repository name as metadata. Re-indexing uses upsert, so the same chunk ID updates its record instead of creating a duplicate. Retrieval and similarity search are intentionally deferred to the next step.

To persist a repository's chunks and embeddings, run:

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/repository/index-to-chroma `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"url":"https://github.com/owner/repository","max_chunks":5}' |
	ConvertTo-Json -Depth 5
```

The response reports files and chunks processed, vectors upserted, total vectors in the collection, and embedding dimension. ChromaDB storage remains on disk across backend restarts.

## Dense vector retrieval

`POST /repository/search` embeds a natural-language query with the same `BAAI/bge-small-en-v1.5` model, then asks ChromaDB to compare that query vector with the stored document vectors. ChromaDB returns the nearest chunks and its native L2 distances; the backend does not calculate similarity itself. `top_k` defaults to 5 and is limited to 50. An optional `repository_name` filters results using the metadata saved during indexing.

Storage and retrieval are separate operations: `/repository/index-to-chroma` creates or updates persistent records, while `/repository/search` embeds a query and reads matching records. Dense retrieval is implemented in this step; BM25 and RRF are planned for the next stage. Reranking and answer generation are not included.

Test semantic search with:

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/repository/search `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"query":"Where are user records loaded?","top_k":5,"repository_name":"repository"}' |
	ConvertTo-Json -Depth 6
```

## Frontend

From the project root, install and run the Vite frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually <http://localhost:5173/>. The backend allows requests from Vite's local origins (`localhost:5173` and `127.0.0.1:5173`).