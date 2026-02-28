"""Frontend service: bridges the Cleo service graph to Tauri HUD clients via gRPC."""

import queue
import signal
import threading
from concurrent import futures

import grpc
import structlog

from generated import frontend_pb2, frontend_pb2_grpc
from services.broadcast import BroadcastHub
from services.config import FRONTEND_PORT

log = structlog.get_logger()

PROTOCOL_VERSION = "1.0.0"


class FrontendServiceServicer(frontend_pb2_grpc.FrontendServiceServicer):
    """gRPC servicer that fans out HUD commands to connected Tauri clients."""

    def __init__(self):
        self._hub = BroadcastHub(maxsize=256)
        self._stop_event = threading.Event()
        log.info("frontend_service.init")

    def push_command(self, hud_command: frontend_pb2.HudCommand) -> None:
        """Inject a HUD command to be sent to all connected clients."""
        self._hub.publish(hud_command)

    def SubscribeHudCommands(self, request, context):
        client_id = request.client_id or "anonymous"
        log.info("frontend_service.subscribe", client_id=client_id)

        sid, q = self._hub.subscribe()
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    command = q.get(timeout=1.0)
                    yield command
                except queue.Empty:
                    continue
        finally:
            self._hub.unsubscribe(sid)
            log.info("frontend_service.unsubscribe", client_id=client_id)

    def SendUserAction(self, request, context):
        action_type = request.WhichOneof("action")
        log.info("frontend_service.user_action", action=action_type)
        return frontend_pb2.ActionResponse(ok=True)

    def GetStatus(self, request, context):
        return frontend_pb2.StatusResponse(
            protocol_version=PROTOCOL_VERSION,
            ready=True,
        )

    def shutdown(self) -> None:
        log.info("frontend_service.shutdown")
        self._stop_event.set()


def serve(port: int = FRONTEND_PORT) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    servicer = FrontendServiceServicer()
    frontend_pb2_grpc.add_FrontendServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("frontend_service.started", port=port)

    def _shutdown(signum, frame):
        del frame
        log.info("frontend_service.signal", signal=signum)
        servicer.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()
