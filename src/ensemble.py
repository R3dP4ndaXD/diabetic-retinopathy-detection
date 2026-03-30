from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from src.model import DRModel


class EnsemblePredictor:
    """Inference-only soft-voting ensemble over DRModel checkpoints."""

    def __init__(
        self,
        checkpoint_paths: Sequence[str],
        weights: Sequence[float] | None = None,
        tta_enabled: bool = False,
        tta_runs: int = 5,
    ) -> None:
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must contain at least one checkpoint path.")

        self.models: List[DRModel] = [DRModel.load_from_checkpoint(path) for path in checkpoint_paths]
        self.tta_enabled = tta_enabled
        self.tta_runs = tta_runs

        num_classes = self.models[0].num_classes
        for idx, model in enumerate(self.models):
            if model.num_classes != num_classes:
                raise ValueError(
                    f"All ensemble members must have the same num_classes. "
                    f"Model 0 has {num_classes}, model {idx} has {model.num_classes}."
                )
        self.num_classes = num_classes

        if weights is None:
            self.weights = torch.full((len(self.models),), 1.0 / len(self.models), dtype=torch.float32)
        else:
            if len(weights) != len(self.models):
                raise ValueError(
                    f"weights length ({len(weights)}) must match number of checkpoints ({len(self.models)})."
                )
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weight_sum = float(weights_tensor.sum().item())
            if weight_sum <= 0:
                raise ValueError("Sum of ensemble weights must be > 0.")
            self.weights = weights_tensor / weight_sum

    def to(self, device: torch.device) -> "EnsemblePredictor":
        for model in self.models:
            model.to(device)
            model.eval()
        self.weights = self.weights.to(device)
        return self

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return weighted-average probabilities for a batch."""
        probs_per_model = []

        for model in self.models:
            if self.tta_enabled:
                avg_probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
                for _ in range(self.tta_runs):
                    augmented = model._tta_transform(x)
                    logits = model(augmented)
                    avg_probs += torch.softmax(logits, dim=1)
                avg_probs /= self.tta_runs
                probs_per_model.append(avg_probs)
            else:
                logits = model(x)
                probs_per_model.append(torch.softmax(logits, dim=1))

        stacked = torch.stack(probs_per_model, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
