from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from code_chunking import router as chunking_router
from chroma_service import router as chroma_router
from embedding_service import router as embedding_router
from hybrid_api import router as hybrid_router
from rag_api import router as rag_router
from repository_ingestion import router as repository_router
from retrieval_service import router as retrieval_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(repository_router)
app.include_router(chunking_router)
app.include_router(embedding_router)
app.include_router(chroma_router)
app.include_router(retrieval_router)
app.include_router(hybrid_router)
app.include_router(rag_router)


@app.get("/")
def health_check():
    return {"message": "CodeLens AI backend is running"}