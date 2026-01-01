## Data preprocessing: VAD + segmentation (general architecture)

This project treats *preprocessing* as a series of optional **row-rewriting steps** that operate on metadata tables (CSVs).
The audio files are not rewritten; instead, we store **time offsets** (seconds) inside the CSV so the dataloader can seek/crop correctly.

The two main steps are:

- **VAD (voice activity detection)**: runs on the *full file* (once) and rewrites file-level rows into trimmed/split rows.
- **Segmentation** (optional): expands each (possibly VAD-trimmed) row into multiple fixed-size segment rows.

The key idea is composability:

1. Start from a file-level metadata table (one row per audio file).
2. Apply optional transforms that rewrite rows (e.g., VAD).
3. Optionally expand rows into segments.

This design keeps the pipeline flexible for future VAD models and future segmentation strategies.

---

## Stage 0: file-level metadata rows

Most prep scripts first build a DataFrame where each row describes a single audio file:

- `rel_filepath`: relative path to the audio file (relative to a dataset root)
- `recording_duration`: duration in seconds
- `speaker_id`, `sample_rate`, `split`, etc.

At this point **a row represents the whole file**.

---

## Stage 1 (optional): VAD at the CSV level

### What VAD does

VAD runs on the **full audio file** and produces detected speech regions (timestamps).
Instead of deleting silence from the waveform, we rewrite the CSV row(s) to represent the useful portion(s) of the file.

VAD rewrites rows by adding fields like:

- `vad_start`: start time of the retained region, **absolute seconds in the original file**
- `vad_end`: end time of the retained region, **absolute seconds in the original file**
- `vad_chunk_id`: identifier for the rewritten row (needed because one file may become multiple rows)
- `vad_speech_timestamps`: JSON list of speech intervals `[[start, end], ...]`, **absolute seconds in the original file**

### How start/end are chosen

VAD usually detects multiple speech segments separated by silence/noise.
We then:

1. **Post-process** detection output to reduce common errors:
	- merge speech segments separated by tiny gaps (e.g., brief pauses)
	- drop extremely short speech bursts (often false positives)

2. **Trim leading/trailing silence**:
	- `vad_start` is the start of the first retained speech segment
	- `vad_end` is the end of the last retained speech segment

3. **Optionally split on long internal silence**:
	- If there is a long silence gap inside the file (configurable and duration-dependent), we split the file into multiple *chunks*.
	- Each chunk becomes a new CSV row with its own `vad_start/vad_end`.

This yields:

- 0 rows (if no speech found -> the file is dropped)
- 1 row (typical)
- N rows (if long internal silence triggers splitting)

### What happens to `recording_duration`

After VAD rewrites a row, `recording_duration` represents the **duration of the retained chunk**:

`recording_duration` = `vad_end` - `vad_start`

This is intentional: downstream steps (including segmentation) operate on the retained chunk, not the entire original file.

### Why keep timestamps absolute

We store `vad_start/vad_end` and `vad_speech_timestamps` in **absolute file time** so that:

- The dataloader can seek correctly into the original file.
- Multiple preprocessing stages can compose without ambiguity.
- Future strategies (e.g., speaker diarization, multi-region sampling) can reuse the same convention.

---

## Stage 2 (optional): segmentation

Segmentation expands each *row* into multiple *segment rows*.
It uses a sliding window:

- window length: `segment_duration`
- hop size: `segment_duration - segment_overlap`

Each segment row typically contains:

- `segment_id`: unique segment identifier
- `start_time`, `end_time`: segment boundaries in seconds
- `segment_duration`
- `rel_filepath`, `speaker_id`, … (carried from the original row)

### How segmentation composes with VAD

Segmentation runs **after VAD** and always produces `start_time/end_time` in **absolute seconds in the original file**.

Practical roadmap (what you get) is easiest to understand in cases:

1. **VAD off, segmentation off**
	- One CSV row represents the whole file.
	- No time offsets are stored.

2. **VAD off, segmentation on**
	- One CSV row (file) becomes many segment rows.
	- Segment windows are generated from `0..recording_duration`.
	- Stored times are already absolute because the base offset is 0:
		- `start_time = start_time_within_file`
		- `end_time = end_time_within_file`

3. **VAD on, segmentation off**
	- One file row becomes 0/1/N rows (drop if no speech; split if long internal silence triggers chunking).
	- Each retained row stores `vad_start` and `vad_end` (absolute file seconds).
	- `recording_duration` is rewritten to `vad_end - vad_start`.

4. **VAD on, segmentation on**
	- First, VAD produces trimmed/split chunk rows.
	- Then segmentation generates windows in *chunk-local time* (`0..recording_duration`), but writes them back as absolute file times using the chunk base offset:
		- `start_time_abs = vad_start + start_time_within_chunk`
		- `end_time_abs = vad_start + end_time_within_chunk`

This keeps downstream loading unambiguous: all final segment boundaries are in original-file time.

### Segment skipping for mostly-silent segments

When `vad_speech_timestamps` is present, a segment can be dropped if it contains too little speech.
This is a soft filter intended to remove *mostly silence* segments while still keeping natural short pauses/noise.

Conceptually:

- Compute speech overlap between the segment window and the speech timestamps.
- Compute `silence_ratio = 1 - speech_ratio`.
- Drop the segment if `silence_ratio` exceeds a configured threshold (e.g., `0.80`).

This keeps diversity (some silence/noise remains) but removes segments that are dominated by silence.

---

## Configuration and future changes

### Dependency injection (DI)

VAD is configured via Hydra dependency injection. The config provides a `_target_` and parameters.
This makes it easy to swap VAD implementations later without changing preprocessing scripts.

If a future VAD model is added, it should provide a similar interface:

- `should_apply(split_name: str) -> bool`
- `apply(rows, audio_root, split_name, ...) -> new_rows`

### Timestamp invariants (recommended)

To keep preprocessing steps composable across future changes:

- Keep all times (`vad_*`, `start_time`, `end_time`) in **seconds**.
- Prefer storing times in **absolute original-file time** in the final CSV artifacts.
- If an intermediate representation uses relative times, convert to absolute as the last step of that stage.

### Snapshot/caching

Preparation artifacts are cached based on a config snapshot.
VAD configuration is included in the snapshot keys so changing VAD settings forces regeneration.

