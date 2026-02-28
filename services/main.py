"""Service runtime: starts persistent Cleo services as subprocesses."""

import multiprocessing
import signal
import threading
import time

import grpc
import structlog

from services.config import (
    ASSISTANT_ADDRESS,
    ASSISTANT_PORT,
    COLOR_BLIND_ADDRESS,
    COLOR_BLIND_PORT,
    DATA_ADDRESS,
    DATA_PORT,
    FACE_DETECTION_ADDRESS,
    FACE_DETECTION_PORT,
    FOOD_MACROS_ADDRESS,
    FOOD_MACROS_PORT,
    FRONTEND_ADDRESS,
    FRONTEND_PORT,
    NAVIGATOR_ADDRESS,
    NAVIGATOR_PORT,
    NOTETAKING_ADDRESS,
    NOTETAKING_PORT,
    SAVE_VIDEO_ADDRESS,
    SAVE_VIDEO_PORT,
    SENSOR_ADDRESS,
    SENSOR_PORT,
    TRACK_ITEM_LOCATE_ADDRESS,
    TRACK_ITEM_LOCATE_PORT,
    TRACK_ITEM_REGISTER_ADDRESS,
    TRACK_ITEM_REGISTER_PORT,
    TRANSCRIPTION_ADDRESS,
    TRANSCRIPTION_PORT,
    WEATHER_ADDRESS,
    WEATHER_PORT,
)

log = structlog.get_logger()

_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5
_PROCESS_KILL_TIMEOUT_SECONDS = 2


def _run_sensor_service() -> None:
    from services.media.sensor_service import serve

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


def _run_frontend_service() -> None:
    from services.frontend.service import serve

    serve(port=FRONTEND_PORT)


def _run_video_service() -> None:
    from services.video.service import serve

    serve()


def _run_notetaking_service() -> None:
    from apps.notetaking import serve

    serve(port=NOTETAKING_PORT)


def _run_food_macros_service() -> None:
    from apps.food_macros import serve

    serve(port=FOOD_MACROS_PORT)


def _run_navigator_service() -> None:
    from apps.navigator import serve

    serve(port=NAVIGATOR_PORT)


def _run_face_detection_service() -> None:
    from apps.face_detection import serve

    serve(port=FACE_DETECTION_PORT)


def _run_save_video_service() -> None:
    from apps.save_video import serve

    serve(port=SAVE_VIDEO_PORT)


def _run_weather_service() -> None:
    from apps.weather import serve

    serve(port=WEATHER_PORT)


def _run_track_item_register_service() -> None:
    from apps.track_item import serve_register

    serve_register(port=TRACK_ITEM_REGISTER_PORT)


def _run_track_item_locate_service() -> None:
    from apps.track_item import serve_locate

    serve_locate(port=TRACK_ITEM_LOCATE_PORT)


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


def _stop_processes(processes: list[multiprocessing.Process]) -> None:
    for proc in processes:
        if proc.is_alive():
            log.info("runtime.terminating_process", process=proc.name, pid=proc.pid)
            proc.terminate()

    join_deadline = time.monotonic() + _PROCESS_TERMINATE_TIMEOUT_SECONDS
    for proc in processes:
        remaining = max(0.0, join_deadline - time.monotonic())
        proc.join(timeout=remaining)

    still_running = [proc for proc in processes if proc.is_alive()]
    for proc in still_running:
        log.warning("runtime.force_kill_process", process=proc.name, pid=proc.pid)
        proc.kill()

    kill_deadline = time.monotonic() + _PROCESS_KILL_TIMEOUT_SECONDS
    for proc in still_running:
        remaining = max(0.0, kill_deadline - time.monotonic())
        proc.join(timeout=remaining)

    for proc in processes:
        if proc.is_alive():
            log.error("runtime.process_stuck", process=proc.name, pid=proc.pid)


def main() -> None:
    log.info("runtime.starting")

    processes: list[multiprocessing.Process] = []
    shutdown_event = threading.Event()

    def _start_process(target, name: str) -> multiprocessing.Process:
        proc = multiprocessing.Process(target=target, daemon=True, name=name)
        proc.start()
        processes.append(proc)
        return proc

    def _abort_if_shutdown_requested() -> None:
        if shutdown_event.is_set():
            raise KeyboardInterrupt

    def _start_optional_grpc_service(
        run_fn,
        name: str,
        address: str,
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Start an optional gRPC service and continue on startup failures."""
        _start_process(run_fn, name)
        try:
            _wait_for_grpc(address, timeout=timeout)
            _abort_if_shutdown_requested()
            return True
        except Exception as exc:
            log.warning(
                "runtime.optional_service_unavailable",
                service=name,
                address=address,
                error=str(exc),
            )
            return False

    def _shutdown(signum, frame) -> None:
        del frame
        log.info("runtime.shutdown_signal", signal=signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # Core dependencies first so downstream services do not spam retries.
        _abort_if_shutdown_requested()
        _start_process(_run_sensor_service, "sensor-service")
        _wait_for_grpc(SENSOR_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_data_service, "data-service")
        _wait_for_grpc(DATA_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_assistant_service, "assistant-service")
        _wait_for_grpc(ASSISTANT_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_frontend_service, "frontend-service")
        _wait_for_grpc(FRONTEND_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_transcription_service, "transcription-service")
        _wait_for_grpc(TRANSCRIPTION_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_video_service, "video-service")
        _abort_if_shutdown_requested()

        _start_process(_run_track_item_register_service, "track-item-register-tool")
        _wait_for_grpc(TRACK_ITEM_REGISTER_ADDRESS)
        _abort_if_shutdown_requested()

        _start_process(_run_track_item_locate_service, "track-item-locate-tool")
        _wait_for_grpc(TRACK_ITEM_LOCATE_ADDRESS)
        _abort_if_shutdown_requested()

        _start_optional_grpc_service(
            _run_color_blind_service,
            "color-blind-tool",
            COLOR_BLIND_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_notetaking_service,
            "notetaking-tool",
            NOTETAKING_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_food_macros_service,
            "food-macros-tool",
            FOOD_MACROS_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_navigator_service,
            "navigator-tool",
            NAVIGATOR_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_face_detection_service,
            "face-detection-tool",
            FACE_DETECTION_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_save_video_service,
            "save-video-tool",
            SAVE_VIDEO_ADDRESS,
        )
        _start_optional_grpc_service(
            _run_weather_service,
            "weather-tool",
            WEATHER_ADDRESS,
        )

        log.info("runtime.services_ready")
        shutdown_event.wait()
    except KeyboardInterrupt:
        log.info("runtime.interrupted")
        raise SystemExit(130)
    except Exception as exc:
        log.exception("runtime.failed", error=str(exc))
        raise
    finally:
        log.info("runtime.stopping")
        _stop_processes(processes)
        log.info("runtime.stopped")


if __name__ == "__main__":
    main()
