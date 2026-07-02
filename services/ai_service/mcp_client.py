from http import HTTPStatus
from typing import Any, Dict, List

import requests

from interfaces.mcp_server import IMCPServer


class MCPServiceClient(IMCPServer):
    def __init__(self, service_url: str):
        self.base_url = service_url

    def list_tools(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/tools")
        return response.json()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[str]:
        response = requests.post(
            f"{self.base_url}/tools/execute",
            json={"tool_name": tool_name, "arguments": arguments},
        )
        if response.status_code != HTTPStatus.OK:
            return [f"Error fecthing from remote MCP service node: {response.text}"]
        return response.json().get("results", [])
