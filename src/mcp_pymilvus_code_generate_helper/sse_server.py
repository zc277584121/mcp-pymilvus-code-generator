import argparse
import logging
from typing import Any, Sequence

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from milvus_connector import MilvusConnector
from starlette.applications import Starlette
from starlette.routing import Route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sse-mcp-pymilvus-code-generate-helper-server")


class McpServer(MilvusConnector):
    def __init__(
        self,
        milvus_uri="http://localhost:19530",
        milvus_token="",
        db_name="default",
    ):
        super().__init__(milvus_uri=milvus_uri, milvus_token=milvus_token, db_name=db_name)
        self.app = Server("mcp-pymilvus-code-generator-server")
        self.setup_tools()

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
                Tool(
                    name="milvus-code-translate-helper",
                    description="Find related documents and code snippets in different programming languages for milvus code translation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A string of Milvus API names in list format to translate from one programming language to another (e.g., ['create_collection', 'insert', 'search'])",
                            },
                            "source_lang": {
                                "type": "string",
                                "description": "Source programming language (e.g., 'python', 'java', 'go', 'csharp', 'node', 'restful')",
                            },
                            "target_lang": {
                                "type": "string",
                                "description": "Target programming language (e.g., 'python', 'java', 'go', 'csharp', 'node', 'restful')",
                            },
                        },
                        "required": ["query", "source_lang", "target_lang"],
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            name = name.replace("_", "-")
            if name == "milvus-pypmilvus-code-generate-helper":
                query = arguments["query"]
                code = await self.pypmilvus_code_generate_helper(query)
                return [TextContent(type="text", text=code)]
            elif name == "milvus-translate-orm-to-milvus-client-code-helper":
                query = arguments["query"]
                code = await self.orm_to_milvus_client_code_translate_helper(query)
                return [TextContent(type="text", text=code)]
            elif name == "milvus-code-translate-helper":
                query = arguments["query"]
                source_lang = arguments["source_lang"]
                target_lang = arguments["target_lang"]
                code = await self.milvus_code_translate_helper(query, source_lang, target_lang)
                return [TextContent(type="text", text=code)]


def create_app(milvus_uri="http://localhost:19530", milvus_token="", db_name="default"):
    server = McpServer(milvus_uri=milvus_uri, milvus_token=milvus_token, db_name=db_name)
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
