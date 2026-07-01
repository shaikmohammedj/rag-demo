import os
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from pydantic import BaseModel


class TOOLNAMES(Enum):
    RETRIEVE_SEMANTIC_CHUNKS = "RETRIEVE_SEMANTIC_CHUNKS"


COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "vector-db-collection")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DB", "./production_chroma.db")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


class ToolExecutionPayload(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


app = FastAPI(title="Distributed MCP Vector DB Service Node")


@app.get("/tools")
def list_tools():
    return [
        {
            "name": "retrieve_semantic_chunks",
            "description": "Performs similarity search across systemic documentation logs.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]


@app.post("/tools/execute")
def execute_tool(payload: ToolExecutionPayload):
    if payload.tool_name != TOOLNAMES.RETRIEVE_SEMANTIC_CHUNKS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Requested tool not supported"
        )

    query = str(payload.arguments.get("query", ""))
    k = int(payload.arguments.get("k", 3))

    results = vector_store.similarity_search(query=query, k=k)
    return {"results": [doc.page_content for doc in results]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
