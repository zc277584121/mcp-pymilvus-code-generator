import argparse
import logging
import os
from typing import Any, Sequence

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from openai import OpenAI
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker
from starlette.applications import Starlette
from starlette.routing import Route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sse-mcp-pymilvus-code-generate-helper-server")


class PymilvusServer:
    def __init__(
        self,
        milvus_uri="http://localhost:19530",
        milvus_token="",
        db_name="default",
    ):
        logger.debug("Initializing PymilvusServer")
        self.milvus_client = MilvusClient(uri=milvus_uri, token=milvus_token, db_name=db_name)

        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.app = Server("mcp-pymilvus-code-generator-server")
        self.setup_tools()

        try:
            self.milvus_client.load_collection("pymilvus_user_guide")
            self.milvus_client.load_collection("mcp_orm")
            self.milvus_client.load_collection("mcp_milvus_client")
        except Exception as e:
            logger.error(f"Fail to load collection: {e}")

    def create_embedding(self, text):
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Fail to create embedding from user query: {e}")
            return None

    def search_similar_documents(self, collection_name, query_text, top_k=10):
        query_embedding = self.create_embedding(query_text)
        if query_embedding is None:
            logger.error("Fail to create embedding from user query. Stop.")
            return []

        try:
            vector_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field="dense",
                param={
                    "metric_type": "IP",
                },
                limit=top_k,
            )

            text_request = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse",
                param={
                    "metric_type": "BM25",
                },
                limit=top_k,
            )

            requests = [vector_request, text_request]

            ranker = WeightedRanker(0.5, 0.5)

            results = self.milvus_client.hybrid_search(
                collection_name=collection_name,
                reqs=requests,
                ranker=ranker,
                limit=top_k,
                output_fields=["metadata", "content"],
            )

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed when searching for similar documents: {e}")
            return []

    async def pypmilvus_code_generate_helper(self, query) -> str:
        """
        Retrieve related pymilvus code/documents for a given query.

        :param query: User query for generating code in natural language
        :return: related pymilvus code/documents for generating code from user query
        """
        results = self.search_similar_documents("pymilvus_user_guide", query)

        if not results:
            logger.warning("No related document found.")
            return "No related document found."

        related_documents = "Here are related pymilvus code/documents found to help you generate code from user query:\n\n"
        for i, hit in enumerate(results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        return related_documents

    async def orm_to_milvus_client_code_translate_helper(self, query) -> str:
        """
        Retrieve related orm and pymilvus client code/documents for a given query.

        :param query: User query for translating orm code to milvus client in natural language
        :return: related orm and pymilvus client code/documents for a user query
        """

        orm_results = self.search_similar_documents("mcp_orm", query)

        if not orm_results:
            logger.warning("No related orm document found.")
            return "No related orm document found."

        milvus_client_results = self.search_similar_documents("mcp_milvus_client", query)

        if not milvus_client_results:
            logger.warning("No related milvus client document found.")
            return "No related milvus client document found."

        related_documents = "Here are related orm and pymilvus client code/documents found to help you translate orm code to milvus client from user query:\n\n"

        related_documents += "Related ORM documents:\n\n"
        for i, hit in enumerate(orm_results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        related_documents += "Related Milvus Client documents:\n\n"
        for i, hit in enumerate(milvus_client_results[0]):
            content = hit["entity"]["content"]
            related_documents += f"{i + 1}:\n{content}\n\n"

        return related_documents

    def setup_tools(self):
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="milvus-pypmilvus-code-generate-helper",
                    description="Find related pymilvus code/documents to help generating code from user input in natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "User query for generating code",
                            }
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="milvus-translate-orm-to-milvus-client-code-helper",
                    description="Find related orm and pymilvus client code/documents to help translating orm code to milvus client from user input in natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "User query for translating orm code to milvus client",
                            }
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            if name == "milvus-pypmilvus-code-generate-helper":
                query = arguments["query"]
                code = await self.pypmilvus_code_generate_helper(query)
                return [TextContent(type="text", text=code)]
            elif name == "milvus-translate-orm-to-milvus-client-code-helper":
                query = arguments["query"]
                code = await self.orm_to_milvus_client_code_translate_helper(query)
                return [TextContent(type="text", text=code)]


def create_app(milvus_uri="http://localhost:19530", milvus_token="", db_name="default"):
    server = PymilvusServer(milvus_uri=milvus_uri, milvus_token=milvus_token, db_name=db_name)
    sse = SseServerTransport("/message")

    class HandleSSE:
        def __init__(self, sse, server):
            self.sse = sse
            self.server = server

        async def __call__(self, scope, receive, send):
            async with self.sse.connect_sse(scope, receive, send) as streams:
                await self.server.app.run(
                    streams[0], streams[1], self.server.app.create_initialization_options()
                )

    class HandleMessages:
        def __init__(self, sse):
            self.sse = sse

        async def __call__(self, scope, receive, send):
            await self.sse.handle_post_message(scope, receive, send)

    routes = [
        Route("/sse", endpoint=HandleSSE(sse, server), methods=["GET"]),
        Route("/message", endpoint=HandleMessages(sse), methods=["POST"]),
    ]

    return Starlette(routes=routes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyMilvus Code Generation Helper (SSE Server)")
    parser.add_argument(
        "--milvus_uri", type=str, default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument("--milvus_token", type=str, default="", help="Milvus server token")
    parser.add_argument("--db_name", type=str, default="default", help="Milvus database name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=23333, help="Port to run the server on")

    args = parser.parse_args()

    app = create_app(
        milvus_uri=args.milvus_uri, milvus_token=args.milvus_token, db_name=args.db_name
    )
    uvicorn.run(app, host=args.host, port=args.port)
