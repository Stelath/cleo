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

    def execute(self, params: dict) -> tuple[bool, str]:
        query = params.get("query", "")
        log.info("object_recognition.execute", query=query)
        # Placeholder — will integrate with camera frame + vision model
        return True, f"Object recognition for: {query} (placeholder — vision integration pending)"


def serve(port: int = OBJECT_RECOGNITION_PORT):
    serve_tool(ObjectRecognitionServicer(), port)


if __name__ == "__main__":
    serve()
