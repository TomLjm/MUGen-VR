from ..common.interfaces import BaseEvaluator, EvalResult


class VBenchAdapter(BaseEvaluator):
    def __init__(self, dims=None):
        self.dimensions = dims or [
            "subject_consistency", "motion_smoothness", "temporal_quality",
            "aesthetic_quality", "imaging_quality",
        ]
        self.evaluator = None
        self.available = None

    def _init_evaluator(self):
        if self.available is not None:
            return
        try:
            from vbench import VBench
            self.evaluator = VBench(self.dimensions)
            self.available = True
        except Exception as exc:
            self.evaluator = None
            self.available = False
            self.import_error = str(exc)

    def evaluate(self, generated_videos, references=None, required=False):
        self._init_evaluator()
        if not self.available:
            if required:
                raise RuntimeError(
                    f"VBench is required but unavailable: {getattr(self, 'import_error', 'unknown error')}"
                )
            return EvalResult(
                metrics={dim: None for dim in self.dimensions},
                details={"status": "skipped", "reason": getattr(self, "import_error", "VBench unavailable")},
            )
        try:
            results = self.evaluator.evaluate(generated_videos)
            metrics = getattr(results, "metrics", results)
            return EvalResult(metrics=metrics, details={"status": "ok"})
        except Exception as exc:
            if required:
                raise RuntimeError(f"VBench evaluation failed: {exc}") from exc
            return EvalResult(
                metrics={dim: None for dim in self.dimensions},
                details={"status": "skipped", "reason": str(exc)},
            )
