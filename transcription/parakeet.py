"""gRPC Transcription Service using NVIDIA Parakeet ASR via NeMo."""

import logging
import signal
import tempfile
import threading
from concurrent import futures

import grpc
import numpy as np
import soundfile as sf
import structlog

from generated import transcription_pb2, transcription_pb2_grpc

log = structlog.get_logger()

_DEFAULT_SAMPLE_RATE = 48000
# Accumulate ~3 seconds of audio before running inference
_ACCUMULATION_SECONDS = 3.0
_QUIET_LOGGERS = ("nemo", "nemo_logger")


def _load_parakeet_model():
    """Load the Parakeet ASR model via NeMo. Called once at startup."""
    import nemo.collections.asr as nemo_asr

    log.info("parakeet.loading_model")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    log.info("parakeet.model_loaded")
    return model


class _QuietNeMoLogs:
    """Temporarily suppress noisy NeMo inference warnings."""

    def __enter__(self):
        self._prior_levels: list[tuple[logging.Logger, int]] = []
        self._nemo_verbosity_cm = None
        for name in _QUIET_LOGGERS:
            logger = logging.getLogger(name)
            self._prior_levels.append((logger, logger.level))
            logger.setLevel(logging.ERROR)
        try:
            from nemo.utils import logging as nemo_logging

            self._nemo_verbosity_cm = nemo_logging.temp_verbosity(nemo_logging.ERROR)
            self._nemo_verbosity_cm.__enter__()
        except Exception:
            self._nemo_verbosity_cm = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._nemo_verbosity_cm is not None:
            self._nemo_verbosity_cm.__exit__(exc_type, exc, tb)
        for logger, level in self._prior_levels:
            logger.setLevel(level)
        return False


class TranscriptionServiceServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC servicer that runs Parakeet ASR inference."""

    def __init__(self):
        self._model = _load_parakeet_model()
        self._inference_lock = threading.Lock()

    def _transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Run inference on an audio buffer. Returns transcribed text.

        NeMo's transcribe() requires file paths, so we write a temp WAV file.
        """
        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with self._inference_lock:
                with _QuietNeMoLogs():
                    try:
                        results = self._model.transcribe([tmp.name], batch_size=1, verbose=False)
                    except TypeError:
                        results = self._model.transcribe([tmp.name])
            # Results may be a list of strings or list of Hypothesis objects
            if results and isinstance(results[0], str):
                return results[0]
            elif results and hasattr(results[0], "text"):
                return results[0].text
            return ""

    def Transcribe(self, request, context):
        """Unary RPC: transcribe a complete audio buffer."""
        sample_rate = request.sample_rate if request.sample_rate > 0 else _DEFAULT_SAMPLE_RATE
        audio = np.frombuffer(request.audio_data, dtype=np.float32)

        if len(audio) == 0:
            return transcription_pb2.TranscriptionResult(text="", confidence=0.0)

        log.info("parakeet.transcribe", samples=len(audio), sample_rate=sample_rate)
        text = self._transcribe_audio(audio, sample_rate)

        duration = len(audio) / sample_rate
        return transcription_pb2.TranscriptionResult(
            text=text,
            confidence=1.0,
            start_time=0.0,
            end_time=duration,
            is_partial=False,
        )

    def TranscribeStream(self, request_iterator, context):
        """Bidirectional streaming: accumulate audio chunks, yield transcription results."""
        sample_rate = _DEFAULT_SAMPLE_RATE
        buffer = []
        buffer_samples = 0
        stream_time = 0.0

        for request in request_iterator:
            if not context.is_active():
                break

            if request.sample_rate > 0:
                sample_rate = request.sample_rate

            chunk = np.frombuffer(request.audio_data, dtype=np.float32)
            buffer.append(chunk)
            buffer_samples += len(chunk)

            threshold_samples = int(_ACCUMULATION_SECONDS * sample_rate)
            should_flush = buffer_samples >= threshold_samples or request.is_final

            if should_flush and buffer_samples > 0:
                audio = np.concatenate(buffer)
                duration = len(audio) / sample_rate
                start_time = stream_time

                log.info(
                    "parakeet.stream_transcribe",
                    samples=len(audio),
                    duration=f"{duration:.2f}s",
                )
                text = self._transcribe_audio(audio, sample_rate)

                stream_time += duration
                buffer = []
                buffer_samples = 0

                yield transcription_pb2.TranscriptionResult(
                    text=text,
                    confidence=1.0,
                    start_time=start_time,
                    end_time=stream_time,
                    is_partial=not request.is_final,
                )

        # Flush any remaining audio
        if buffer_samples > 0:
            audio = np.concatenate(buffer)
            duration = len(audio) / sample_rate
            text = self._transcribe_audio(audio, sample_rate)
            yield transcription_pb2.TranscriptionResult(
                text=text,
                confidence=1.0,
                start_time=stream_time,
                end_time=stream_time + duration,
                is_partial=False,
            )


def serve(port: int = 50052):
    """Start the transcription gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ],
    )
    servicer = TranscriptionServiceServicer()
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("parakeet.started", port=port)

    def _shutdown(signum, frame):
        log.info("parakeet.stopping", signal=signum)
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
