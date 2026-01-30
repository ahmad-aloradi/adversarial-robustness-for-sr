---
created: 2026-01-30T10:09
title: Fix CNCeleb FLAC decoding errors in prep script
area: tooling
files:
  - scripts/datasets/prep_cnceleb.sh
---

## Problem

The `prep_cnceleb.sh` script fails during Stage 3 (Convert FLAC to WAV) with FLAC decoding errors:

```
[flac @ 0x57b5b94550c0] invalid residual
[flac @ 0x57b5b94550c0] decode_frame() failed
[aist#0:0/flac @ 0x57b5b9427b40] Decoding error: Invalid data found when processing input
```

This occurs when processing the CNCeleb dataset (669,292 FLAC files). The errors indicate corrupted or invalid FLAC files in the dataset, possibly from:
1. Incomplete/corrupted archive extraction
2. Corrupted source files in the original dataset
3. Issues with the combined CN-Celeb2 parts (`cat cn-celeb2_v2.tar.gz.*`)

The script uses ffmpeg for conversion:
```bash
ffmpeg -i "$flac_file" -acodec pcm_s16le -ar 16000 "$wav_file" -y -loglevel error
```

## Solution

TBD - Investigate:

1. Check if extraction completed properly (look for truncated files)
2. Add error handling to skip/log corrupted files instead of failing silently
3. Consider adding a verification step after extraction
4. May need to re-download corrupted archive parts
5. Add `|| true` or proper error handling to continue past bad files while logging them
