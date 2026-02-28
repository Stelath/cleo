"""Navigation assistance tool service."""

import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from core.config import NAVIGATION_ASSIST_PORT

log = structlog.get_logger()


class NavigationAssistServicer(ToolServiceBase):
    """Helps the user navigate and find directions."""

    @property
    def tool_name(self) -> str:
        return "navigation_assist"

    def execute(self, params: dict) -> tuple[bool, str]:
        query = params.get("query", "")
        log.info("navigation_assist.execute", query=query)
        # Placeholder — will integrate with location services
        return True, f"Navigation for: {query} (placeholder — location integration pending)"


def serve(port: int = NAVIGATION_ASSIST_PORT):
    serve_tool(NavigationAssistServicer(), port)


if __name__ == "__main__":
    serve()
