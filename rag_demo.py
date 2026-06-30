#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain>=0.3.0",
#     "langchain-community>=0.3.0",
#     "langchain-ollama>=0.2.0",
#     "chromadb>=0.5.0",
#     "neo4j>=5.20.0",
#     "fastapi>=0.111.0",
#     "uvicorn>=0.30.0",
# ]
# ///

"""
This is an academic excursion / learning into building a simple RAG based AI system.
There are 4 pieces
    - DataIngestion: This class ingests data and persists it into a vector and a graph database.
    - VectorMCPServer: This class exposes a tool call to get related chunks from the vector db.
    - GraphMCPServer: Gets a knowledge graph from Neo4j graph db.
    - AIService: Ties the above components into a service for downstream consumption.
"""

import glob
import os
import sys
from typing import Any

from fastapi import FastAPI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from pydantic import BaseModel


class DataIngestionEngine:
    def __init__(self, chroma_db_dir="./chroma.db"):
        self.chroma_db_dir = chroma_db_dir
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, length_function=len
        )
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)

    def run_pipeline(self, file_name: str, content: str):
        docs = self.text_splitter.create_documents(
            texts=[content], metadatas=[{"source": file_name}]
        )
        Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.chroma_db_dir,
        )
        print(
            f"[Vector database] Indexed {len(docs)} chunks locally using nomic-embed-text"
        )

        query = """
        MERGE (s:Component {name: "PaymentService"})
        MERGE (d: Component {name: AuthDatabase})
        MERGE (t:Asset {from: AuthTokens})

        MERGE (s)-[:DEPENDS_ON]->(d)
        MERGE (d)-[:STORES]->(t)
        """

        with self.neo4j_driver.session() as sess:
            sess.run(query)
        print("[Graph Layer] Injected architectural relationships into Neo4j")

    def close(self):
        self.neo4j_driver.close()


class MCPVectorServer:
    def __init__(self, chroma_db_dir="./chroma.db"):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.db = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

    def call_tool(self, tool_name: str, args: dict[str, Any]):
        if tool_name == "retrieve_semantic_chunks":
            results = self.db.similarity_search(
                args.get("query", ""), k=args.get("k", 2)
            )
            return [doc.page_content for doc in results]
        raise ValueError("Error matching valid tool")


class MCPGraphServer:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)

    def call_tool(self, tool_name: str, args: dict):
        if tool_name == "retrieve_graph_relations":
            query = """
            MATCH (c:Component {name: $entity})-[r1-DEPENDS_ON]->(target)-[r2:STORES]->(asset)
            RETURN c.name AS source, target.name as dependency, asset.name as asset_stored
            """

            relations = []
            with self.driver.session() as sess:
                result = sess.run(query, entity=args.get("entity"))
                for record in result:
                    relations.append(
                        f"Infrastructure Topology: Component [{record['source']}] DEPENDS on [{record['dependency']}] which STORES data asset [{record['asset_stored']}]."
                    )
            return relations if relations else ["No explicit dependencies stored."]
        raise ValueError("Error matching valid tool")


class AIService:
    def __init__(self, vector_mcp: MCPVectorServer, graph_mcp: MCPGraphServer):
        self.vector_mcp = vector_mcp
        self.graph_mcp = graph_mcp
        self.llm = ChatOllama(model="llama3", temperature=0.1)

    def query_rag(self, user_query: str, target_entity: str):
        text_context = self.vector_mcp.call_tool(
            "retrieve_semantic_chunks", {"query": user_query}
        )
        graph_context = self.graph_mcp.call_tool(
            "retrieve_graph_relations", {"entity": target_entity}
        )

        system_prompt = f"""
        You are senior system infrastructure assistant.
        Synthesize a clean analysis given the context fragments.

        SEMANTIC DOCUMENT SEGMENTS:
        {" ".join(text_context)}

        CONNECTED INFRASTRUCTURE GRAPH MAPPING:
        {" ".join(graph_context)}

        Provide the root cause for the issue.
        """

        messages = [("system", system_prompt), ("human", user_query)]
        return self.llm.invoke(messages).content


app = FastAPI(title="Test RAG system")

vector_server = MCPVectorServer()
graph_server = MCPGraphServer()
ai_service = AIService(vector_mcp=vector_server, graph_mcp=graph_server)


class QueryRequest(BaseModel):
    query: str
    focus_entity: str


@app.post("/query")
def process_system_query(payload: QueryRequest):
    answer = ai_service.query_rag(
        user_query=payload.query, target_entity=payload.focus_entity
    )
    return {"status": "success", "analysis": answer}


def bootstrap_from_docs():
    documents_glob = os.path.join("docs", "*.txt")
    documents = glob.glob(documents_glob)

    if not documents:
        print("[ISSUE] No files in the docs/ directory.")
        sys.exit(1)

    ingestor = DataIngestionEngine()
    try:
        for doc in documents:
            file_name = os.path.basename(doc)
            with open(doc, "r", encoding="utf-8") as f:
                content = f.read()
            ingestor.run_pipeline(file_name, content)
    finally:
        ingestor.close()


if __name__ == "__main__":
    import uvicorn

    if "--seed" in sys.argv:
        bootstrap_from_docs()

    print("Starting API service...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
