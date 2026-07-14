"""Modality dropout for robustness training.

Innovation C: improves robustness when some modalities are missing at inference.
"""
import torch
import torch.nn as nn


class ModalityDropout(nn.Module):
    """Randomly drop modalities during training for robustness."""

    def __init__(self, dropout_prob=0.2, num_modalities=3):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_modalities = num_modalities
        # _training is inherited from nn.Module via .training property

    def forward(self, modality_dict):
        """
        Args:
            modality_dict: dict like {"text": emb, "image": emb, "audio": emb}
        Returns:
            dict with some modalities zeroed out
        """
        if not self.training:
            return modality_dict

        available = [k for k, v in modality_dict.items() if v is not None]
        if len(available) <= 1:
            return modality_dict

        result = dict(modality_dict)
        for mod in available:
            if torch.rand(1).item() < self.dropout_prob:
                result[mod] = None

        # Ensure at least one modality remains
        remaining = [k for k, v in result.items() if v is not None]
        if len(remaining) == 0:
            result[available[0]] = modality_dict[available[0]]

        return result
