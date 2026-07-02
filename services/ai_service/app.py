import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus

from fastapi import FastAPI, HTTPException, logger
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel

from services.ai_service.mcp_client import MCPServiceClient

app = FastAPI(title="AI Gateway orchestration service")

VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://localhost:8001")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8002")

vector_mcp_client = MCPServiceClient(VECTOR_SERVICE_URL)
graph_mcp_client = MCPServiceClient(GRAPH_SERVICE_URL)

llm = ChatOllama(model="llama3", temperature=0.1)


class Payload(BaseModel):
    prompt: str
    context_target_entity: str


@app.post("/v1/query")
def orchestrate_hybrid_rag(payload: Payload):
    tool_calls = [
        (
            vector_mcp_client.call_tool,
            {
                "tool_name": "retrieve_semantic_chunks",
                "arguments": {"query": payload.prompt},
            },
        ),
        (
            graph_mcp_client.call_tool,
            {
                "tool_name": "retrieve_graph_relations",
                "arguments": {"query": payload.context_target_entity},
            },
        ),
    ]

    results = []
    try:
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(fn, **args) for fn, args in tool_calls]
            for task in as_completed(tasks):
                results.append(task.result())
    except Exception as e:
        logger.logger.error(e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="error orchestrating responses",
        )

    vector_data, graph_data, _ = results
    system_prompt = f"""
    You are an operations intelligence bot.
        Synthesize a post-mortem analysis using the decoupled platform inputs below.

        SYSTEM MATRIX CHUNKS (VECTOR DB LAYER):
        {" ".join(vector_data)}

        TOPOLOGY GRAPH RELATIONSHIPS (GRAPH DB LAYER):
        {" ".join(graph_data)}
    """

    messages = [("system", system_prompt), ("human", payload.prompt)]
    response = llm.invoke(messages)

    return {"status": "success", "model": "llama3-local", "response": response.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
