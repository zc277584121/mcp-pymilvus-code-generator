import argparse
import asyncio
import logging
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from milvus_connector import MilvusConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stdio-mcp-pymilvus-code-generate-server")


def main():
    parser = argparse.ArgumentParser(description="PyMilvus Code Generation Helper")
    parser.add_argument(
        "--milvus_uri", type=str, default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument("--milvus_token", type=str, default="", help="Milvus server token")
    parser.add_argument("--db_name", type=str, default="default", help="Milvus database name")

    args = parser.parse_args()

    pymilvus_server = MilvusConnector(
        milvus_uri=args.milvus_uri, milvus_token=args.milvus_token, db_name=args.db_name
    )

    server = Server("stdio-mcp-pymilvus-code-generate-helper-server")

    @server.list_tools()
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

    @server.call_tool()
    async def call_tool(
        name: str, arguments: Any
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        name = name.replace("_", "-")
        if name == "milvus-pypmilvus-code-generate-helper":
            query = arguments["query"]
            code = await pymilvus_server.pypmilvus_code_generate_helper(query)
            return [TextContent(type="text", text=code)]
        elif name == "milvus-translate-orm-to-milvus-client-code-helper":
            query = arguments["query"]
            code = await pymilvus_server.orm_to_milvus_client_code_translate_helper(query)
            return [TextContent(type="text", text=code)]
        elif name == "milvus-code-translate-helper":
            query = arguments["query"]
            source_lang = arguments["source_lang"]
            target_lang = arguments["target_lang"]
            code = await pymilvus_server.milvus_code_translate_helper(
                query, source_lang, target_lang
            )
            return [TextContent(type="text", text=code)]

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server running with stdio transport")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
