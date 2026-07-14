#!/usr/bin/env python3
"""Download, trim, validate, and index MSR-VTT clips without redistributing media."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mugen.data.manifest import read_jsonl, sha256_file, write_jsonl


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe(path: Path):
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
            "-of", "json", str(path),
        ]
    )
    payload = json.loads(result.stdout)
    stream_types = {stream.get("codec_type") for stream in payload.get("streams", [])}
    return float(payload["format"]["duration"]), stream_types


def prepare(row, root: Path, overwrite: bool):
    split_dir = root / row["split"]
    video_path = split_dir / "video" / f"{row['video_id']}.mp4"
    audio_path = split_dir / "audio" / f"{row['video_id']}.wav"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    output = dict(row)
    try:
        if overwrite or not video_path.exists():
            section = f"*{row['start']}-{row['end']}"
            run(
                [
                    "yt-dlp", "--no-playlist", "--download-sections", section,
                    "--force-keyframes-at-cuts", "-S", "res:360,ext:mp4:m4a",
                    "--recode-video", "mp4", "-o", str(video_path), row["url"],
                ]
            )
        run([
            "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn", "-ac", "1",
            "-ar", "16000", str(audio_path),
        ])
        duration, streams = probe(video_path)
        if "video" not in streams or "audio" not in streams:
            raise ValueError(f"required streams missing: {sorted(streams)}")
        output.update(
            {
                "status": "ok",
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "decoded_duration": duration,
                "media_sha256": sha256_file(video_path),
            }
        )
    except Exception as exc:
        output.update({"status": "failed", "reason": str(exc)[:1000]})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="data/msrvtt/media")
    parser.add_argument("--result", default="data/msrvtt/media_manifest.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-valid", type=int, default=5000)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, row, Path(args.output_root), args.overwrite) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 25 == 0:
                write_jsonl(results, args.result)
                print({"processed": index, "valid": sum(row["status"] == "ok" for row in results)})
    write_jsonl(results, args.result)
    valid = sum(row["status"] == "ok" for row in results)
    print({"processed": len(results), "valid": valid, "failed": len(results) - valid})
    if not args.limit and valid < args.min_valid:
        raise RuntimeError(f"only {valid} valid clips; release requires at least {args.min_valid}")


if __name__ == "__main__":
    main()
