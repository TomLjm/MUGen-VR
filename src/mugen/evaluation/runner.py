import os
from ..common.interfaces import BaseEvaluator, EvalResult
from ..common.utils import save_json

class EvaluationRunner(BaseEvaluator):
    def __init__(self, output_dir="./outputs/eval"):
        self.evaluators = {}
        self.output_dir = output_dir

    def register(self, name, evaluator):
        self.evaluators[name] = evaluator

    def evaluate(self, generated_videos, references=None):
        all_metrics = {}
        all_details = {}
        for name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(generated_videos, references)
            all_metrics[name] = result.metrics
            all_details[name] = result.details
        return EvalResult(metrics=all_metrics, details=all_details)

    def run_full_eval(self, videos, references=None, save_report=True):
        result = self.evaluate(videos, references)
        if save_report:
            os.makedirs(self.output_dir, exist_ok=True)
            save_json({"metrics": result.metrics, "details": result.details},
                       os.path.join(self.output_dir, "eval_results.json"))
        return result
