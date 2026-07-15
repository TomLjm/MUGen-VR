#!/usr/bin/env python3
"""Build the static Hugging Face Space from verified project evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_gates(gates):
    if gates is None:
        return []
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim < 1:
        return []
    return values.reshape(-1, values.shape[-1]).mean(axis=0).tolist()


def build_space(ablation_rows, report, cases, output_dir, existing_results):
    output_dir = Path(output_dir)
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    by_key = {(str(row["pair_id"]), row["variant"]): row for row in ablation_rows}
    built_cases = []
    for case in cases:
        pair_id = str(case["pair_id"])
        b0 = by_key.get((pair_id, "B0"))
        b3 = by_key.get((pair_id, "B3"))
        if b0 is None or b3 is None:
            raise ValueError(f"missing B0/B3 generated pair for showcase case: {pair_id}")
        safe_id = "".join(character if character.isalnum() else "-" for character in pair_id)
        relative_b0 = Path("assets") / f"{safe_id}-b0.mp4"
        relative_b3 = Path("assets") / f"{safe_id}-b3.mp4"
        shutil.copy2(b0["generated_video_path"], output_dir / relative_b0)
        shutil.copy2(b3["generated_video_path"], output_dir / relative_b3)
        built_cases.append(
            {
                "pair_id": pair_id,
                "caption": case["caption"],
                "generation_seed": case["generation_seed"],
                "b0_video": relative_b0.as_posix(),
                "b3_video": relative_b3.as_posix(),
                "b0_seconds": b0["metrics"]["latency_seconds"],
                "b3_seconds": b3["metrics"]["latency_seconds"],
                "references": [item["sample_id"] for item in b3.get("references", [])],
                "gates": summarize_gates(b3.get("gates")),
                "note": "Fixed held-out pair; identical generation seed.",
            }
        )
    metrics = [
        {"variant": variant, **report["summary"][variant]["means"]}
        for variant in ("B0", "B1", "B2", "B3")
    ]
    payload = {
        **existing_results,
        "status": "held-out-evaluation-complete",
        "metrics": metrics,
        "cases": built_cases,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Build static MUGen Hugging Face Space")
    parser.add_argument("--ablation-input", required=True)
    parser.add_argument("--final-report", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--output-dir", default="hf_space")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = build_space(
        load_jsonl(args.ablation_input),
        json.loads(Path(args.final_report).read_text(encoding="utf-8")),
        load_jsonl(args.case_manifest),
        output_dir,
        json.loads((output_dir / "results.json").read_text(encoding="utf-8")),
    )
    print(json.dumps({"output": str(output_dir), "cases": len(payload["cases"])}))


if __name__ == "__main__":
    main()
