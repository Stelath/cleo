"""Frontend service: bridges the Cleo service graph to Tauri HUD clients via gRPC."""

import queue
import signal
import threading
from concurrent import futures

import grpc
import structlog

from generated import frontend_pb2, frontend_pb2_grpc
from services.media.broadcast import BroadcastHub
from services.config import FRONTEND_PORT
from services.frontend.voice import ElevenLabsVoice

log = structlog.get_logger()

PROTOCOL_VERSION = "1.0.0"


class FrontendServiceServicer(frontend_pb2_grpc.FrontendServiceServicer):
    """gRPC servicer with typed RPCs that fan out DisplayUpdates to Tauri clients."""

    def __init__(self):
        self._hub = BroadcastHub(maxsize=256)
        self._stop_event = threading.Event()
        self._voice = ElevenLabsVoice()
        log.info("frontend_service.init")

    def _publish_update(self, **kwargs) -> frontend_pb2.FrontendResponse:
        """Wrap a typed request in a DisplayUpdate and publish to all subscribers."""
        update = frontend_pb2.DisplayUpdate(**kwargs)
        self._hub.publish(update)
        return frontend_pb2.FrontendResponse(ok=True)

    # ── Typed push RPCs ──

    def ShowNotification(self, request, context):
        log.info("frontend_service.show_notification", title=request.title)
        return self._publish_update(notification=request)

    def StreamImage(self, request_iterator, context):
        expected_index = 0
        image_id = None

        for chunk in request_iterator:
            if image_id is None:
                image_id = chunk.image_id
                expected_index = 0

            if chunk.image_id != image_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("StreamImage image_id changed mid-stream")
                return frontend_pb2.FrontendResponse(ok=False)

            if chunk.chunk_index != expected_index:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"StreamImage chunk gap: expected {expected_index}, got {chunk.chunk_index}"
                )
                return frontend_pb2.FrontendResponse(ok=False)

            self._publish_update(image_chunk=chunk)
            expected_index += 1

            if chunk.is_last:
                break

        log.info("frontend_service.stream_image", image_id=image_id)
        return frontend_pb2.FrontendResponse(ok=True)

    def ShowProgress(self, request, context):
        log.info("frontend_service.show_progress", label=request.label, value=request.value)
        return self._publish_update(progress=request)

    def ShowText(self, request, context):
        log.info("frontend_service.show_text")
        return self._publish_update(text=request)

    def ShowCard(self, request, context):
        log.info("frontend_service.show_card", count=len(request.cards))
        return self._publish_update(card=request)

    def Clear(self, request, context):
        log.info("frontend_service.clear")
        return self._publish_update(clear=request)

    def PlayAudio(self, request, context):
        log.info("frontend_service.play_audio", sample_rate=request.sample_rate)
        return self._publish_update(play_audio=request)

    def PlayAudioFile(self, request, context):
        log.info("frontend_service.play_audio_file", path=request.path)
        return self._publish_update(play_audio_file=request)

    def SpeakText(self, request, context):
        text = request.text
        if not text or not text.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("SpeakText requires non-empty text")
            return frontend_pb2.FrontendResponse(ok=False)

        try:
            data_base64, sample_rate = self._voice.synthesize(text)
        except Exception as exc:
            log.error("frontend_service.speak_text_failed", error=str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"TTS synthesis failed: {exc}")
            return frontend_pb2.FrontendResponse(ok=False)

        log.info("frontend_service.speak_text", text_len=len(text))
        play_req = frontend_pb2.PlayAudioRequest(
            data_base64=data_base64,
            sample_rate=sample_rate,
        )
        return self._publish_update(play_audio=play_req)

    def RenderHtml(self, request, context):
        log.info("frontend_service.render_html")
        return self._publish_update(render_html=request)

    def ShowThrobber(self, request, context):
        log.info("frontend_service.show_throbber", visible=request.visible)
        return self._publish_update(throbber=request)

    # ── Streaming (Tauri subscribes here) ──

    def StreamUpdates(self, request, context):
        client_id = request.client_id or "anonymous"
        log.info("frontend_service.stream_updates", client_id=client_id)

        sid, q = self._hub.subscribe()
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    update = q.get(timeout=1.0)
                    yield update
                except queue.Empty:
                    continue
        finally:
            self._hub.unsubscribe(sid)
            log.info("frontend_service.stream_ended", client_id=client_id)

    # ── User actions & status (unchanged) ──

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
