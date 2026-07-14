"""Dataset manifest construction and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


REQUIRED_COLUMNS = {"video_id", "caption", "url", "start time", "end time", "split"}
SPLIT_ALIASES = {"validate": "val", "validation": "val", "valid": "val"}


def normalize_split(value: str) -> str:
    split = SPLIT_ALIASES.get(str(value).lower(), str(value).lower())
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {value}")
    return split


def load_msrvtt_metadata(paths: Iterable[str | Path]) -> List[Dict]:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise ValueError("at least one parquet metadata path is required")
    frame = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"MSR-VTT metadata is missing columns: {sorted(missing)}")
    records = []
    for row in frame.to_dict("records"):
        start = float(row["start time"])
        end = float(row["end time"])
        if end <= start:
            raise ValueError(f"invalid clip interval for {row['video_id']}: {start}..{end}")
        video_id = str(row["video_id"])
        records.append(
            {
                "sample_id": f"{normalize_split(row['split'])}:{video_id}",
                "video_id": video_id,
                "caption": str(row["caption"]),
                "url": str(row["url"]),
                "start": start,
                "end": end,
                "duration": end - start,
                "split": normalize_split(row["split"]),
            }
        )
    validate_split_isolation(records)
    return records


def validate_split_isolation(records: Iterable[Dict]) -> None:
    assignments: Dict[str, str] = {}
    for row in records:
        video_id = str(row["video_id"])
        split = normalize_split(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"video_id {video_id} appears in both {previous} and {split}")


def write_jsonl(records: Iterable[Dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str | Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
