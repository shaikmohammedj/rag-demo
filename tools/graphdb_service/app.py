import os
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel


class TOOLNAMES(Enum):
    RETRIEVE_GRAPH_RELATIONS = "retrieve_graph_relations"


app = FastAPI(title="MCP GraphDB service")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7867")
NEO4J_AUTH = os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", "")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


class RequestPayload(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@app.get("/tools")
def list_tools():
    return [
        {
            "name": "retrieve_graph_relations",
            "description": "Traces multi-hop dependency topologies inside Neo4j.",
            "input_schema": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
            },
        }
    ]


@app.post("/tools/execute")
def execute_tool(payload: RequestPayload):
    if payload.tool_name != TOOLNAMES.RETRIEVE_GRAPH_RELATIONS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="invalid tool call"
        )

    entity = payload.arguments.get("entity", "")
    query = """
    MATCH (c:Component {name:$entity})-[r1:DEPENDS_ON]->(target)-[r2:STORES]->(asset)
    RETURN c.name AS source, target.name AS dependency, asset.name AS asset_stored
    """

    relations = []
    with driver.session() as sess:
        result = sess.run(query, entity=entity)
        for rec in result:
            relations.append(
                f"Infrastructre State: Component [{rec['source']}] DEPENDS ON Database [{rec['dependency']}] which HOUSES asset [{rec['asset_stored']}]."
            )

    return {
        "results": relations if relations else ["No cross boundry graph relations."]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8082)
