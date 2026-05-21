"""Optional FastAPI app."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.main import HallucinationAwareMultimodalRAG


class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="Hallucination-Aware Multimodal RAG")
rag = HallucinationAwareMultimodalRAG()


@app.post("/run")
def run_query(request: QueryRequest) -> dict:
    result = rag.run(image=None, query=request.query)
    return result.dict()
