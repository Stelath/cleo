"""Navigation assistance tool service."""

import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from services.config import NAVIGATION_ASSIST_PORT

log = structlog.get_logger()


class NavigationAssistServicer(ToolServiceBase):
    """Helps the user navigate and find directions."""

    @property
    def tool_name(self) -> str:
        return "navigation_assist"

    @property
    def tool_description(self) -> str:
        return (
            "Help the user navigate or find directions. Use when the user asks "
            "for directions, nearby places, or help getting somewhere."
        )

    @property
    def tool_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Navigation request or destination",
                },
            },
            "required": ["query"],
        }

    def execute(self, params: dict) -> tuple[bool, str]:
        query = params.get("query", "")
        log.info("navigation_assist.execute", query=query)
        # Placeholder — will integrate with location services
        return True, f"Navigation for: {query} (placeholder — location integration pending)"


def serve(port: int = NAVIGATION_ASSIST_PORT):
    serve_tool(NavigationAssistServicer(), port)


if __name__ == "__main__":
    serve()
