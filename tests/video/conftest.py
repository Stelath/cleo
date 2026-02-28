"""Fixtures for video embedding integration tests.

All Bedrock fixtures require BEDROCK_TESTS=1 and valid AWS credentials.
"""

import os
from concurrent import futures

import grpc
import pytest

from generated import data_pb2, data_pb2_grpc
from services.video.service import VideoDataClient

requires_bedrock = pytest.mark.skipif(
    os.getenv("BEDROCK_TESTS") != "1",
    reason="Set BEDROCK_TESTS=1 to run Bedrock integration tests",
)

_MAX_MSG = 48 * 1024 * 1024  # 48 MB — request carries mp4_data + embed_data (~38 MB)


@pytest.fixture
def data_server(tmp_path):
    """Start a real DataServiceServicer on an ephemeral gRPC port.

    Uses tmp_path for db, index, and video storage so tests are isolated.
    Yields (server, port) and shuts down on teardown.
    """
    from services.data.service import DataServiceServicer

    servicer = DataServiceServicer(
        db_path=str(tmp_path / "cleo.db"),
        index_path=str(tmp_path / "clips.index"),
        video_dir=str(tmp_path / "videos"),
        tracked_item_dir=str(tmp_path / "tracked_items"),
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    data_pb2_grpc.add_DataServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    yield server, port

    servicer.shutdown()
    server.stop(grace=2)


@pytest.fixture
def video_data_client(data_server):
    """VideoDataClient pointed at the ephemeral data_server with raised message limit."""
    _, port = data_server
    client = VideoDataClient.__new__(VideoDataClient)
    client._timeout = 120.0
    client._channel = grpc.insecure_channel(
        f"localhost:{port}",
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    client._stub = data_pb2_grpc.DataServiceStub(client._channel)
    yield client
    client.close()


@pytest.fixture
def data_stub(data_server):
    """Raw DataServiceStub for direct Search RPCs."""
    _, port = data_server
    channel = grpc.insecure_channel(
        f"localhost:{port}",
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    stub = data_pb2_grpc.DataServiceStub(channel)
    yield stub
    channel.close()
