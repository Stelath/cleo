# Parakeet Transcription Service Upgrade Plan

## Goal

Upgrade the current `transcription/parakeet.py` service to deliver:

- Lower end-to-end latency for partial transcripts
- Higher final-text accuracy
- Better continuity across chunk boundaries by reusing prior audio/text context
- Cleaner semantics for partial vs final transcript updates

## Current State Summary

The current implementation is simple but it is not truly streaming:

- `TranscribeStream()` accumulates about 3 seconds of audio before each inference (`_ACCUMULATION_SECONDS = 3.0`).
- Each flush runs a fresh `model.transcribe()` call on only the buffered window.
- Audio is written to a temporary WAV file for every inference because the current path uses `ASRModel.transcribe()` with file inputs.
- There is no overlapping audio window, no decoder state reuse, and no text stabilization logic.
- `is_partial=True` currently means "not final yet", but those partials are complete independent transcripts of each flushed window, not revisions of one rolling utterance.
- A single `_inference_lock` serializes all requests, so multiple streams cannot decode concurrently.

This means the service loses linguistic context at every flush boundary and will tend to:

- Miss words split across chunk boundaries
- Produce unstable phrasing between consecutive partials
- Delay first useful text until the 3-second threshold is reached
- Waste time on repeated temp-file I/O

## Main Problems To Solve

### 1. Latency is too high

The first transcript can only appear after enough audio reaches the 3-second threshold. With the current bridge settings (`500 ms` chunks from `core/main.py`), the user may wait multiple chunks before seeing any text.

### 2. Context is discarded between inferences

Each window is decoded independently. The model never sees the prior acoustic tail or the already-decoded text, so accuracy drops at boundaries.

### 3. Partial results are not revision-friendly

The downstream bridge appends partial strings together. That works for non-overlapping independent chunks, but it becomes incorrect once the service starts emitting improved revisions of the same utterance.

### 4. The inference path is inefficient

Writing a WAV file for every decode adds avoidable overhead and increases jitter.

### 5. The protocol is too minimal for real streaming ASR

`AudioInput` and `TranscriptionResult` only carry raw audio, timestamps, and `is_final`. There is no way to identify an utterance, replace prior partials, or describe transcript stability.

## Target Architecture

Move from "micro-batch transcription" to "incremental utterance decoding":

1. Continuously ingest small audio chunks (for example 80 to 200 ms).
2. Maintain per-stream session state in memory.
3. Build rolling decode windows with overlap, not disjoint windows.
4. Emit low-latency partial hypotheses frequently.
5. Keep a committed prefix plus a revisable suffix.
6. Finalize an utterance on endpointing (silence, explicit final chunk, or timeout).

At a high level, each active stream should maintain:

- A ring buffer of recent audio
- A current utterance start time
- A committed transcript prefix
- A latest unstable suffix
- Decoder metadata needed to compare new hypotheses with prior ones

## Recommended Phases

## Phase 1: Fix the Streaming Contract

Before changing the decoder, make the RPC contract capable of supporting transcript revisions.

### Proto changes

Extend `protos/transcription.proto` so responses can describe updates to the same utterance rather than only append-only text.

Recommended additions:

- Add `string utterance_id` to `TranscriptionResult`
- Add `uint32 revision` to `TranscriptionResult`
- Add `string committed_text` to `TranscriptionResult`
- Add `string unstable_text` to `TranscriptionResult`
- Add `float stability` to `TranscriptionResult`
- Optionally add `bool replaces_previous` if you want explicit replacement semantics

Optional request-side additions:

- Add `string stream_id` to `AudioInput` so the service can support explicit session identity across reconnects
- Add `bool reset_context` to force a hard utterance boundary

### Consumer changes

Update `core/main.py` so `AudioTranscriptionBridge` stops concatenating partial strings blindly.

New downstream behavior:

- For partial results: cache by `utterance_id`, replacing older revisions
- For final results: persist only the final committed text for that utterance
- If partial persistence is desired later, store revisions separately rather than folding them into the final transcript row

This change is necessary before overlap-based decoding, because future partials will intentionally revise prior text.

## Phase 2: Reduce Latency Without Breaking Accuracy

Change the streaming cadence first, even before full context-aware decoding.

### Buffering changes

Replace the fixed 3-second batch trigger with two windows:

- `emit_window`: small interval for partial updates, target `160 to 320 ms`
- `decode_window`: rolling context window for inference, target `1.5 to 4.0 s`

Practical starting point:

- Ingest chunks at `100 ms`
- Emit partials every `200 ms`
- Decode the last `2.0 to 2.5 s` of audio
- Keep `300 to 500 ms` of overlap beyond the last committed boundary

### Endpointing changes

The current silence detection in `core/main.py` is coarse (`500 ms` chunking and `1000 ms` final silence). Tighten this because endpoint delay directly impacts final-result latency.

Recommended starting point:

- Lower `_AUDIO_CHUNK_MS` from `500` to `100` or `160`
- Revisit `_FINAL_SILENCE_MS` to something closer to `400 to 700`
- Keep a minimum speech duration threshold to avoid finalizing on noise bursts

This alone should make the service feel substantially faster.

## Phase 3: Add Context Across Incoming Audio

This is the key step for your stated requirement: use previous audio to improve the new audio as it comes in.

### Rolling overlapping decode

Instead of decoding only new audio, decode a rolling window that includes:

- The most recent undecided audio
- A short overlap from the previously decoded region

Example:

- At time `t`, decode audio from `t - 2.5 s` to `t`
- On the next emit, decode from `t - 2.3 s` to `t + 0.2 s`

Then align the new hypothesis with the previous one and:

- Commit the stable common prefix
- Keep the trailing words unstable until they persist across multiple revisions

This is the standard way to improve streaming ASR quality when the model itself is not exposing persistent decoder state.

### Prefix-stability algorithm

Implement a lightweight hypothesis reconciliation layer:

1. Normalize previous and current text (trim, collapse whitespace, optionally lowercase for comparison only).
2. Compute the longest stable prefix between the previous hypothesis and the new one.
3. Only mark words as committed after they survive `N` consecutive revisions or move outside the overlap zone.
4. Emit:
   - `committed_text`: stable prefix
   - `unstable_text`: latest revisable suffix

This gives downstream consumers stable text early without locking in bad boundary guesses too soon.

### Why this helps

Even if the model is still called as a stateless decoder, overlapping windows let the new decode "hear" a small amount of prior context, which improves:

- Word segmentation across chunk boundaries
- Proper nouns and short function words near transitions
- Punctuation and phrase completion

## Phase 4: Remove Temp WAV I/O

The current `_transcribe_audio()` path writes every buffer to disk and calls `transcribe()` on a file path. That is functional but not ideal for low-latency streaming.

### Preferred direction

Switch to an in-memory inference path supported by the NeMo model/runtime you are using. The exact implementation depends on what `parakeet-tdt-0.6b-v3` exposes in your installed NeMo version, but the preferred order is:

1. Use a direct tensor/audio-array inference API if available
2. Use a lower-level decoding API that accepts features or tensors
3. Keep file-based fallback only for compatibility

Implementation target:

- Convert float32 PCM bytes to a CUDA-ready tensor in memory
- Resample only if required by the model
- Run inference without touching disk

If NeMo does not expose a clean direct API for this model, build the rest of the streaming/session logic first and keep the temp-file path behind an interface so it can be swapped later.

## Phase 5: Introduce Per-Stream Session State

Right now, `TranscribeStream()` uses local variables only for one iterator pass. Make that state explicit and structured.

### Add a session object

Create a private class or dataclass such as `StreamingSession` that owns:

- `sample_rate`
- `utterance_id`
- `stream_time`
- `audio_ring_buffer`
- `last_emit_time`
- `committed_text`
- `latest_hypothesis`
- `revision`
- `speech_active`
- `last_voice_time`

This makes the code easier to reason about and is the natural place to add:

- Overlap management
- Hypothesis reconciliation
- Endpointing logic
- Metrics (latency, decode duration, revision count)

### Concurrency model

Revisit `_inference_lock`.

Today it guarantees safety but also forces one decode at a time for the entire service. Depending on actual GPU throughput, choose one of these:

- Keep single-flight inference but make it explicit in the design, with a bounded request queue and metrics
- Allow one session per GPU decode slot with a semaphore
- Move decode work to a dedicated worker thread that batches compatible requests

For an initial upgrade, single-flight plus better buffering is acceptable, but the service should measure queue delay so you know when it becomes the bottleneck.

## Phase 6: Improve Accuracy Beyond Raw Decoding

Once the stream mechanics are fixed, layer on targeted accuracy improvements.

### Resampling and normalization

Verify the model’s expected sample rate. The service defaults to `48000`, while the bridge uses `16000`.

Upgrade path:

- Make the expected model sample rate explicit
- Resample input consistently inside the service if the source rate differs
- Normalize or validate amplitude ranges so the model sees predictable input

Do not rely on callers to always send the ideal rate.

### Voice activity handling

The bridge currently handles endpointing before the ASR service. That is fine for a first pass, but longer term you will get better control by making endpointing partly ASR-aware.

Recommended direction:

- Keep upstream VAD for coarse gating
- Add service-side utterance timeout and minimum trailing silence checks
- Optionally add a small pre-roll buffer so the first phoneme is not clipped

### Optional language-model rescoring

If Parakeet/NeMo exposes beam outputs or n-best hypotheses in your stack, consider a second-pass rescoring step for final transcripts only. Keep it off the partial path unless the latency cost is negligible.

### Domain biasing

If this platform has domain-specific vocabulary (people, product names, commands), add a configurable biasing layer for final transcript correction. That can be:

- Prompt/context biasing if the decoder supports it
- Phrase list post-correction
- Named-entity replacement against a known lexicon

## Phase 7: Instrumentation and Validation

Do not ship streaming changes without measuring them.

### Add metrics

Record at least:

- Time from first speech chunk to first partial
- Time from final speech chunk to final transcript
- Decode duration per inference call
- Queue wait before decode
- Partial revision count per utterance
- Average committed-text growth per revision

### Add logs

Add structured logs around:

- Utterance start/end
- Partial emitted
- Final emitted
- Decode window size
- Overlap size
- Stability score

### Add tests

Add unit tests for:

- Rolling buffer window construction
- Prefix reconciliation
- Partial-to-final replacement semantics
- Endpointing thresholds

Add integration tests for:

- A phrase split across two adjacent chunks
- A long utterance with multiple partial revisions
- Silence-triggered finalization
- Mixed sample-rate input

## Suggested Implementation Order

1. Update the protobuf contract to support utterance IDs and revisions.
2. Update `AudioTranscriptionBridge` to treat partials as replaceable state, not append-only text.
3. Lower audio chunk size in `core/main.py` and replace the 3-second flush rule with a faster emit cadence.
4. Introduce a `StreamingSession` abstraction inside `transcription/parakeet.py`.
5. Implement rolling overlapping decode plus stable-prefix reconciliation.
6. Swap temp-file inference for in-memory decoding if NeMo supports it cleanly.
7. Tune thresholds using real recordings and latency metrics.

## Concrete Code Targets

Primary files to change:

- `transcription/parakeet.py`
- `core/main.py`
- `protos/transcription.proto`
- `generated/transcription_pb2.py` and `generated/transcription_pb2_grpc.py` after regeneration

Likely internal refactors in `transcription/parakeet.py`:

- Split model loading from stream session logic
- Add a decoder adapter layer so inference transport (temp WAV vs in-memory tensor) is isolated
- Replace the `buffer`/`buffer_samples` locals in `TranscribeStream()` with session state
- Replace direct `yield text` semantics with revision-aware result objects

## Risks and Tradeoffs

- Smaller windows reduce latency but can increase unstable partials if overlap is too short.
- Longer rolling windows improve context but increase GPU time per update.
- Frequent partial updates improve responsiveness but can overwhelm downstream consumers unless revision semantics are clear.
- In-memory decoding may require model-specific NeMo internals, which can increase maintenance cost.

The right balance is usually:

- Fast, slightly unstable partials
- Slower, high-confidence committed text
- Very clear replacement semantics for downstream consumers

## Practical First Milestone

If you want the highest impact with the lowest implementation risk, start with this milestone:

1. Cut audio chunks to `100 to 160 ms`.
2. Emit partials every `200 to 300 ms`.
3. Decode a rolling `2 to 3 s` window with `400 ms` overlap.
4. Reconcile new text against the prior hypothesis and emit `committed_text + unstable_text`.
5. Finalize after `500 to 700 ms` of silence.

That will materially reduce latency and improve boundary accuracy even before deeper NeMo-specific optimization work.

## Success Criteria

The upgrade is successful when:

- First partial text appears in under `500 ms` after speech begins
- Final transcript appears in under `800 ms` after speech ends
- Words split across chunk boundaries are preserved reliably
- Partial revisions become progressively more stable instead of producing disconnected 3-second fragments
- Downstream persistence stores one coherent final utterance per speech segment
