#!/usr/bin/env python3
"""Merge generation, retrieval, audio-control, and VBench outputs for final acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("B0", "B1", "B2", "B3")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl_files(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def keyed(rows, variants=None):
    result = {}
    for row in rows:
        key = (str(row["pair_id"]), row["variant"])
        if variants is not None and row["variant"] not in variants:
            raise ValueError(f"unexpected variant in metrics: {row['variant']}")
        if key in result:
            raise ValueError(f"duplicate metric row: {key}")
        result[key] = row
    return result


def merge_metrics(generation_rows, retrieval_payload, audio_payload, vbench_payloads):
    generation = keyed(generation_rows, VARIANTS)
    audio_rows = [
        {**row, "pair_id": row.get("pair_id", row.get("sample_id"))}
        for row in audio_payload["per_sample"]
    ]
    audio = keyed(audio_rows, VARIANTS)
    retrieval = keyed(retrieval_payload["per_sample"], ("B2", "B3"))
    if set(audio) != set(generation):
        raise ValueError("audio-control rows do not exactly cover generated rows")
    expected_retrieval = {key for key in generation if key[1] in {"B2", "B3"}}
    if set(retrieval) != expected_retrieval:
        raise ValueError("retrieval rows do not exactly cover B2/B3 generated rows")

    vbench_metrics = {}
    for variant in VARIANTS:
        payload = vbench_payloads[variant]
        if payload["vbench"]["details"].get("status") != "ok":
            raise ValueError(f"VBench did not complete for {variant}")
        metrics = payload["vbench"]["metrics"]
        if metrics.get("vbench_total") is None:
            raise ValueError(f"VBench total is missing for {variant}")
        vbench_metrics[variant] = metrics

    merged = []
    for key in sorted(generation):
        row = generation[key]
        audio_row = audio[key]
        metrics = {
            **row["metrics"],
            **vbench_metrics[row["variant"]],
            "imagebind_audio_video_alignment": audio_row["imagebind_audio_video_alignment"],
            "onset_flow_correlation": audio_row["onset_flow_correlation"],
        }
        if row["variant"] in {"B2", "B3"}:
            retrieval_row = retrieval[key]
            metrics["retrieval_rank"] = retrieval_row["retrieval_rank"]
            metrics["retrieval_mrr"] = retrieval_row["retrieval_reciprocal_rank"]
        merged.append({**row, "metrics": metrics})
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Merge strict MUGen project metrics")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--vbench-root", required=True)
    parser.add_argument("--output", default="reports/project/ablation-input.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    generation_paths = sorted(Path(args.generation_dir).glob("results-part-*.jsonl"))
    if not generation_paths:
        raise ValueError("no partitioned generation results found")
    vbench_payloads = {
        variant: load_json(Path(args.vbench_root) / variant / "demo_report.json")
        for variant in VARIANTS
    }
    rows = merge_metrics(
        load_jsonl_files(generation_paths),
        load_json(args.retrieval),
        load_json(args.audio),
        vbench_payloads,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
