"""Exercise the notetaking Bedrock summary path and optional note persistence.

This script is intended as a manual smoke test for the notetaking service.
It can:

1. Build a synthetic note context from CLI-provided transcripts and optional MP4s.
2. Load a real note context from DataService for a time window.
3. Call the same summary logic used by NotetakingServicer.
4. Optionally persist the generated summary and fetch recent note summaries back.

Examples:
    uv run python scripts/test_notetaking_service.py
    uv run python scripts/test_notetaking_service.py --transcript "We reviewed the roadmap."
    uv run python scripts/test_notetaking_service.py --last-seconds 120 --store
    uv run python scripts/test_notetaking_service.py --transcript "Inspect this clip" --mp4 clip.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from apps.notetaking import NoteContext, NotetakingServicer
from generated import data_pb2
from services.config import DATA_ADDRESS

_DEFAULT_TRANSCRIPTS = [
    "We reviewed the launch checklist and agreed to fix the camera latency issue first.",
    "The team decided to run one more hardware validation pass before shipping.",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test the notetaking service summary flow against Bedrock and optionally "
            "store the resulting note in DataService."
        ),
    )
    parser.add_argument(
        "--data-address",
        default=DATA_ADDRESS,
        help="DataService gRPC address used for range fetches and note storage.",
    )
    parser.add_argument(
        "--transcript",
        action="append",
        default=[],
        help=(
            "Transcript text to include in a synthetic context. Repeat for multiple entries. "
            "If omitted and no range is requested, built-in sample transcripts are used."
        ),
    )
    parser.add_argument(
        "--mp4",
        action="append",
        default=[],
        help="Path to an MP4 clip to include in a synthetic multimodal context. Repeat as needed.",
    )
    parser.add_argument(
        "--start-ts",
        type=float,
        help="Start timestamp (epoch seconds) for loading a real context from DataService.",
    )
    parser.add_argument(
        "--end-ts",
        type=float,
        default=None,
        help="End timestamp (epoch seconds) for loading a real context from DataService.",
    )
    parser.add_argument(
        "--last-seconds",
        type=float,
        help="Load a real context for the last N seconds from now.",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Persist the generated summary via DataService.StoreNoteSummary and fetch recent notes.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=5,
        help="Number of note summaries to fetch back after storing.",
    )
    return parser.parse_args()


def _resolve_window(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.last_seconds is not None:
        end_ts = time.time() if args.end_ts is None else args.end_ts
        return end_ts - args.last_seconds, end_ts

    if args.start_ts is not None:
        end_ts = time.time() if args.end_ts is None else args.end_ts
        return args.start_ts, end_ts

    return None


def _build_synthetic_context(
    *,
    transcripts: list[str],
    mp4_paths: list[str],
    start_ts: float,
    end_ts: float,
) -> NoteContext:
    transcript_entries: list[data_pb2.TranscriptionLogEntry] = []
    transcript_texts = transcripts or list(_DEFAULT_TRANSCRIPTS)

    total_duration = max(end_ts - start_ts, 1.0)
    per_entry = total_duration / max(len(transcript_texts), 1)
    for index, text in enumerate(transcript_texts):
        entry_start = start_ts + (index * per_entry)
        entry_end = min(end_ts, entry_start + per_entry)
        transcript_entries.append(
            data_pb2.TranscriptionLogEntry(
                id=index + 1,
                text=text,
                confidence=0.99,
                start_time=entry_start,
                end_time=entry_end,
                created_at=time.time(),
            )
        )

    clips: list[tuple[data_pb2.VideoClipMetadata, data_pb2.VideoClipResponse]] = []
    for index, raw_path in enumerate(mp4_paths):
        path = Path(raw_path)
        mp4_data = path.read_bytes()
        clip_start = start_ts + (index * 0.5)
        clip_end = min(end_ts, clip_start + 5.0)
        metadata = data_pb2.VideoClipMetadata(
            clip_id=index + 1,
            clip_path=str(path),
            start_timestamp=clip_start,
            end_timestamp=clip_end,
            num_frames=0,
            created_at=time.time(),
        )
        clip = data_pb2.VideoClipResponse(
            mp4_data=mp4_data,
            start_timestamp=clip_start,
            end_timestamp=clip_end,
            num_frames=0,
        )
        clips.append((metadata, clip))

    return NoteContext(transcripts=transcript_entries, clips=clips)


def _store_and_fetch(
    *,
    servicer: NotetakingServicer,
    summary_text: str,
    start_ts: float,
    end_ts: float,
    list_limit: int,
) -> int:
    stored = servicer._data.StoreNoteSummary(
        data_pb2.StoreNoteSummaryRequest(
            summary_text=summary_text,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )
    )
    print(f"Stored note summary with id={stored.id}")

    recent = servicer._data.GetNoteSummaries(
        data_pb2.NoteSummariesRequest(limit=list_limit, offset=0)
    )
    print(f"Fetched {len(recent.entries)} recent note summaries (total={recent.total_count})")
    for entry in recent.entries:
        marker = " <== inserted" if entry.id == stored.id else ""
        print(
            f"- id={entry.id} window=({entry.start_timestamp:.3f}, {entry.end_timestamp:.3f})"
            f"{marker}"
        )
        print(f"  {entry.summary_text}")

    return stored.id


def main() -> int:
    args = _parse_args()
    time_window = _resolve_window(args)
    end_ts = time.time() if args.end_ts is None else args.end_ts
    start_ts = (end_ts - 30.0) if time_window is None else time_window[0]
    end_ts = end_ts if time_window is None else time_window[1]

    servicer = NotetakingServicer(data_address=args.data_address)
    try:
        if time_window is not None:
            print(
                "Loading note context from DataService "
                f"for {start_ts:.3f} -> {end_ts:.3f} at {args.data_address}"
            )
            context = servicer._load_note_context(start_ts, end_ts)
        else:
            print("Building synthetic note context from CLI inputs.")
            context = _build_synthetic_context(
                transcripts=args.transcript,
                mp4_paths=args.mp4,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        print(
            f"Context ready: transcripts={len(context.transcripts)} clips={len(context.clips)} "
            f"window=({start_ts:.3f}, {end_ts:.3f})"
        )

        if not context.transcripts and not context.clips:
            print("No content found. Bedrock will not be called for an empty context.")
            return 1

        summary_text = servicer._summarize_context(start_ts, end_ts, context)
        print("\nGenerated summary:\n")
        print(summary_text)

        if args.store:
            print("")
            _store_and_fetch(
                servicer=servicer,
                summary_text=summary_text,
                start_ts=start_ts,
                end_ts=end_ts,
                list_limit=args.list_limit,
            )

        return 0
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        servicer.close()


if __name__ == "__main__":
    raise SystemExit(main())
