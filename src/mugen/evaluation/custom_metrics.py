import torch
from ..common.interfaces import BaseEvaluator, EvalResult


class CustomMetrics(BaseEvaluator):
    def evaluate(self, generated_videos, references=None):
        metrics = {}
        details = {"status": "ok"}
        if isinstance(generated_videos, torch.Tensor):
            metrics.update(self.compute_temporal_metrics(generated_videos))
        if references and "gen_features" in references and "ref_features" in references:
            metrics.update(self.compute_alignment(references["gen_features"], references["ref_features"]))
        if references and all(k in references for k in ["query_embs", "gallery_embs", "labels"]):
            metrics.update(self.compute_retrieval_metrics(references["query_embs"], references["gallery_embs"], references["labels"]))
        if not metrics:
            details = {"status": "skipped", "reason": "no compatible tensors supplied"}
        return EvalResult(metrics=metrics, details=details)

    def compute_retrieval_metrics(self, query_embs, gallery_embs, labels, ks=(1, 5, 10)):
        query_embs = torch.nn.functional.normalize(query_embs.float(), dim=-1)
        gallery_embs = torch.nn.functional.normalize(gallery_embs.float(), dim=-1)
        scores = torch.mm(query_embs, gallery_embs.T)
        results = {}
        labels = labels.to(scores.device)
        for k in ks:
            kk = min(k, gallery_embs.shape[0])
            _, indices = torch.topk(scores, kk, dim=1)
            recall = (indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"retrieval_recall@{k}"] = recall
        ranks = torch.argsort(scores, dim=1, descending=True)
        rr = []
        for row, label in zip(ranks, labels):
            pos = (row == label).nonzero(as_tuple=False)
            rr.append(1.0 / (pos[0].item() + 1) if len(pos) else 0.0)
        results["retrieval_mrr"] = float(sum(rr) / len(rr)) if rr else 0.0
        return results

    def compute_alignment(self, gen_features, ref_features):
        cos = torch.nn.CosineSimilarity(dim=-1)
        scores = cos(gen_features.float(), ref_features.float())
        return {"alignment_cosine": scores.mean().item(), "alignment_std": scores.std(unbiased=False).item()}

    def compute_temporal_metrics(self, video):
        if video.dim() == 5:
            video = video[0]
        if video.shape[0] < 2:
            return {"temporal_delta": 0.0, "temporal_variance": 0.0}
        diffs = (video[1:].float() - video[:-1].float()).abs().mean(dim=(1, 2, 3))
        return {"temporal_delta": diffs.mean().item(), "temporal_variance": diffs.var(unbiased=False).item()}
