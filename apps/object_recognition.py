"""Object recognition tool service."""

import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from services.config import OBJECT_RECOGNITION_PORT

log = structlog.get_logger()


class ObjectRecognitionServicer(ToolServiceBase):
    """Identifies and describes objects in the user's view."""

    @property
    def tool_name(self) -> str:
        return "object_recognition"

    @property
    def tool_description(self) -> str:
        return (
            "Identify and describe objects in the user's current view. Use when "
            "the user asks 'what is this?', 'what am I looking at?', or wants "
            "to identify something in front of them."
        )

    @property
    def tool_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants identified or described",
                },
            },
            "required": ["query"],
        }

    def execute(self, params: dict) -> tuple[bool, str]:
        query = params.get("query", "")
        log.info("object_recognition.execute", query=query)
        # Placeholder — will integrate with camera frame + vision model
        return True, f"Object recognition for: {query} (placeholder — vision integration pending)"


def serve(port: int = OBJECT_RECOGNITION_PORT):
    serve_tool(ObjectRecognitionServicer(), port)


if __name__ == "__main__":
    serve()
