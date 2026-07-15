from __future__ import annotations

import json
import sys
from pathlib import Path

from ..common.interfaces import BaseEvaluator, EvalResult


class VBenchAdapter(BaseEvaluator):
    """Run the vendored VBench source without installing its pinned dependencies."""

    def __init__(self, dims=None, output_path="reports/vbench"):
        self.dimensions = dims or [
            "subject_consistency",
            "motion_smoothness",
            "temporal_flickering",
            "aesthetic_quality",
            "imaging_quality",
        ]
        self.output_path = Path(output_path)
        self.evaluator = None
        self.available = None

    def _load_vbench_class(self):
        source_path = Path(__file__).resolve().parents[3] / "third_party" / "VBench"
        if source_path.is_dir() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        from vbench import VBench

        return VBench, source_path

    def _init_evaluator(self):
        if self.available is not None:
            return
        try:
            import torch

            vbench_class, source_path = self._load_vbench_class()
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.evaluator = vbench_class(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                str(source_path / "VBench_full_info.json"),
                str(self.output_path),
            )
            self.available = True
        except Exception as exc:
            self.evaluator = None
            self.available = False
            self.import_error = str(exc)

    @staticmethod
    def _dimension_score(value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, dict):
            value = value.get("score", value.get("value"))
        return float(value)

    def evaluate(self, generated_videos, references=None, required=False):
        del references
        self._init_evaluator()
        metric_names = [*self.dimensions, "vbench_total"]
        if not self.available:
            if required:
                raise RuntimeError(
                    f"VBench is required but unavailable: {getattr(self, 'import_error', 'unknown error')}"
                )
            return EvalResult(
                metrics={dim: None for dim in metric_names},
                details={"status": "skipped", "reason": getattr(self, "import_error", "VBench unavailable")},
            )
        try:
            name = "mugen_eval"
            self.evaluator.evaluate(
                videos_path=str(generated_videos),
                name=name,
                prompt_list=[],
                dimension_list=self.dimensions,
                mode="custom_input",
            )
            result_path = self.output_path / f"{name}_eval_results.json"
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = {
                dimension: self._dimension_score(raw[dimension])
                for dimension in self.dimensions
            }
            metrics["vbench_total"] = sum(metrics.values()) / len(metrics)
            return EvalResult(
                metrics=metrics,
                details={"status": "ok", "result_path": str(result_path)},
            )
        except Exception as exc:
            if required:
                raise RuntimeError(f"VBench evaluation failed: {exc}") from exc
            return EvalResult(
                metrics={dim: None for dim in metric_names},
                details={"status": "skipped", "reason": str(exc)},
            )
