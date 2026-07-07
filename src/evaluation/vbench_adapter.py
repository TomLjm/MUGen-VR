from ..common.interfaces import BaseEvaluator, EvalResult

class VBenchAdapter(BaseEvaluator):
    def __init__(self, dims=None):
        self.dimensions = dims or [
            "subject_consistency", "motion_smoothness", "temporal_quality",
            "aesthetic_quality", "imaging_quality",
        ]
        self.evaluator = None

    def _init_evaluator(self):
        try:
            from vbench import VBench
            self.evaluator = VBench(self.dimensions)
        except ImportError:
            print("VBench not installed.")

    def evaluate(self, generated_videos, references=None):
        if self.evaluator is None:
            self._init_evaluator()
        if self.evaluator is None:
            return EvalResult(metrics={dim: 0.0 for dim in self.dimensions},
                              details={"note": "VBench not available"})
        try:
            results = self.evaluator.evaluate(generated_videos)
            return EvalResult(metrics=results.metrics, details={})
        except Exception as e:
            return EvalResult(metrics={dim: 0.0 for dim in self.dimensions},
                              details={"error": str(e)})
