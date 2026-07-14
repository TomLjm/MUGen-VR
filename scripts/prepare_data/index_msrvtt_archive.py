#!/usr/bin/env python3
"""Validate a directly hosted MSR-VTT archive and extract 16 kHz audio."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mugen.data.manifest import read_jsonl, sha256_file, write_jsonl


def run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(detail[-4000:]) from exc


def inspect(row, videos_by_id, audio_root: Path, keyframe_root: Path, overwrite: bool):
    output = dict(row)
    video_path = videos_by_id.get(row["video_id"])
    if video_path is None:
        return {**output, "status": "failed", "reason": "video missing from archive"}
    audio_path = audio_root / row["split"] / f"{row['video_id']}.wav"
    keyframe_path = keyframe_root / row["split"] / f"{row['video_id']}.jpg"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    keyframe_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe = run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
            "-of", "json", str(video_path),
        ])
        payload = json.loads(probe.stdout)
        streams = {stream.get("codec_type") for stream in payload.get("streams", [])}
        if "video" not in streams or "audio" not in streams:
            raise ValueError(f"required streams missing: {sorted(streams)}")
        if overwrite or not audio_path.exists():
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn", "-ac", "1",
                "-ar", "16000", str(audio_path),
            ])
        if overwrite or not keyframe_path.exists():
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-frames:v", "1",
                "-q:v", "2", str(keyframe_path),
            ])
        output.update(
            {
                "status": "ok",
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "keyframe_path": str(keyframe_path),
                "decoded_duration": float(payload["format"]["duration"]),
                "media_sha256": sha256_file(video_path),
            }
        )
    except Exception as exc:
        output.update({"status": "failed", "reason": str(exc)[:4000]})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--audio-root", default="data/msrvtt/audio16k")
    parser.add_argument("--keyframe-root", default="data/msrvtt/keyframes")
    parser.add_argument("--result", default="data/msrvtt/media_manifest.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-valid", type=int, default=5000)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    videos = list(Path(args.video_root).rglob("*.mp4"))
    videos_by_id = {path.stem: path for path in videos}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                inspect,
                row,
                videos_by_id,
                Path(args.audio_root),
                Path(args.keyframe_root),
                args.overwrite,
            )
            for row in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0:
                write_jsonl(results, args.result)
                print({"processed": index, "valid": sum(row["status"] == "ok" for row in results)})
    results.sort(key=lambda row: row["sample_id"])
    write_jsonl(results, args.result)
    valid = sum(row["status"] == "ok" for row in results)
    print({"archive_videos": len(videos), "processed": len(results), "valid": valid})
    if not args.limit and valid < args.min_valid:
        raise RuntimeError(f"only {valid} valid clips; release requires at least {args.min_valid}")


if __name__ == "__main__":
    main()
