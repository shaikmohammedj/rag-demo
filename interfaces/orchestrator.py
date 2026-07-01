from typing import Any, Dict, Protocol


class IOrchestrator(Protocol):
    def execute_workflow(self, user_query: str, paramaters: Dict[str, Any]) -> str: ...
