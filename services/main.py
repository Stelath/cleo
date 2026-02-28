"""Service runtime: starts persistent Cleo services as subprocesses."""

import multiprocessing
import signal
import threading

import grpc
import structlog

from services.config import (
    ASSISTANT_ADDRESS,
    ASSISTANT_PORT,
    COLOR_BLIND_ADDRESS,
    COLOR_BLIND_PORT,
    DATA_ADDRESS,
    DATA_PORT,
    NAVIGATION_ASSIST_ADDRESS,
    NAVIGATION_ASSIST_PORT,
    NOTETAKING_ADDRESS,
    NOTETAKING_PORT,
    OBJECT_RECOGNITION_ADDRESS,
    OBJECT_RECOGNITION_PORT,
    SENSOR_ADDRESS,
    SENSOR_PORT,
    TRANSCRIPTION_ADDRESS,
    TRANSCRIPTION_PORT,
)

log = structlog.get_logger()


def _run_sensor_service() -> None:
    from services.sensor_service import serve

    serve(port=SENSOR_PORT)


def _run_transcription_service() -> None:
    from services.transcription.service import serve

    serve(port=TRANSCRIPTION_PORT)


def _run_data_service() -> None:
    from services.data.service import serve

    serve(port=DATA_PORT)


def _run_assistant_service() -> None:
    from services.assistant.service import serve

    serve(port=ASSISTANT_PORT)


def _run_color_blind_service() -> None:
    from apps.color_blind import serve

    serve(port=COLOR_BLIND_PORT)


def _run_object_recognition_service() -> None:
    from apps.object_recognition import serve

    serve(port=OBJECT_RECOGNITION_PORT)


def _run_navigation_assist_service() -> None:
    from apps.navigation_assist import serve

    serve(port=NAVIGATION_ASSIST_PORT)


def _run_video_service() -> None:
    from services.video.service import serve

    serve()
def _run_notetaking_service() -> None:
    from apps.notetaking import serve

    serve(port=NOTETAKING_PORT)


def _wait_for_grpc(address: str, timeout: float = 30.0) -> None:
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        log.info("runtime.grpc_ready", address=address)
    except grpc.FutureTimeoutError as exc:
        log.error("runtime.grpc_timeout", address=address, timeout=timeout)
        raise RuntimeError(
            f"gRPC server at {address} did not become ready in {timeout}s"
        ) from exc
    finally:
        channel.close()


def main() -> None:
    log.info("runtime.starting")

    processes = [
        multiprocessing.Process(target=_run_sensor_service, daemon=True, name="sensor-service"),
        multiprocessing.Process(target=_run_data_service, daemon=True, name="data-service"),
        multiprocessing.Process(target=_run_assistant_service, daemon=True, name="assistant-service"),
        multiprocessing.Process(
            target=_run_transcription_service,
            daemon=True,
            name="transcription-service",
        ),
        multiprocessing.Process(target=_run_color_blind_service, daemon=True, name="color-blind-tool"),
        multiprocessing.Process(
            target=_run_object_recognition_service,
            daemon=True,
            name="object-recognition-tool",
        ),
        multiprocessing.Process(
            target=_run_navigation_assist_service,
            daemon=True,
            name="navigation-assist-tool",
        ),
        multiprocessing.Process(
            target=_run_video_service,
            daemon=True,
            name="video-service",
        ),
        multiprocessing.Process(target=_run_notetaking_service, daemon=True, name="notetaking-tool"),
    ]

    for proc in processes:
        proc.start()

    _wait_for_grpc(SENSOR_ADDRESS)
    _wait_for_grpc(DATA_ADDRESS)
    _wait_for_grpc(ASSISTANT_ADDRESS)
    _wait_for_grpc(TRANSCRIPTION_ADDRESS)
    _wait_for_grpc(COLOR_BLIND_ADDRESS)
    _wait_for_grpc(OBJECT_RECOGNITION_ADDRESS)
    _wait_for_grpc(NAVIGATION_ASSIST_ADDRESS)
    _wait_for_grpc(NOTETAKING_ADDRESS)
    log.info("runtime.services_ready")

    shutdown_event = threading.Event()

    def _shutdown(signum, frame) -> None:
        del frame
        log.info("runtime.shutdown_signal", signal=signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()

    log.info("runtime.stopping")
    for proc in processes:
        proc.terminate()
    for proc in processes:
        proc.join(timeout=5)
    log.info("runtime.stopped")


if __name__ == "__main__":
    main()
