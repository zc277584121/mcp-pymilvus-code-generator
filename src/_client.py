import asyncio
import sys
from contextlib import AsyncExitStack
from typing import Optional

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUERY_PROMPT = """
Use your tools to retrieve the most relevant information from the given query.

Query:
{query}
"""


class MCPClient:
    def __init__(
        self,
        server_script_path: str,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1000,
    ):
        # Initialize session and client objects
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.model_name = model_name
        self.max_tokens = max_tokens

    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = self.server_script_path.endswith(".py")
        if not is_python:
            raise ValueError("Server script must be a .py file")

        command = sys.executable
        server_params = StdioServerParameters(
            command=command, args=[self.server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def aretrieve(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": QUERY_PROMPT.format(query=query)}]

        response = await self.session.list_tools()
        available_tools = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
            for tool in response.tools
        ]

        response = self.anthropic.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=available_tools,
        )

        retrieved_file_names = []
        for content in response.content:
            if content.type == "text":
                pass  # TODO: the tool using is not triggered, consider how to handle it
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                result = await self.session.call_tool(tool_name, tool_args)
                # TODO: transform the result to the format of retrieved file names
                # retrieved_file_names.append(file_name)

        return  # retrieved_file_names

    def retrieve(self, query: str):
        return asyncio.run(self.aretrieve(query))
