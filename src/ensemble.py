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

        self.models: List[DRModel] = []
        for path in checkpoint_paths:
            model = DRModel.load_from_checkpoint(path)
            # Use Lightning's built-in freeze to completely detach gradients and set eval()
            model.freeze()
            self.models.append(model)
            
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
        self.weights = self.weights.to(device)
        return self

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return weighted-average probabilities for a batch."""
        # Dynamically ensure weights are on the exact same device as the input tensor
        self.weights = self.weights.to(x.device)
        
        probs_per_model = []

        for model in self.models:
            if self.tta_enabled:
                # 1. Generate all augmented views
                augmented_views = [model._tta_transform(x) for _ in range(self.tta_runs)]
                
                # 2. Concatenate into one massive batch (Shape: [tta_runs * batch_size, C, H, W])
                super_batch = torch.cat(augmented_views, dim=0)
                
                # 3. Perform a single, highly parallelized forward pass
                super_logits = model(super_batch)
                super_probs = torch.softmax(super_logits, dim=1)
                
                # 4. Reshape back to [tta_runs, batch_size, num_classes] and average the views
                avg_probs = super_probs.view(self.tta_runs, x.size(0), self.num_classes).mean(dim=0)
                
                probs_per_model.append(avg_probs)
            else:
                logits = model(x)
                probs_per_model.append(torch.softmax(logits, dim=1))

        # Shape: [num_models, batch_size, num_classes]
        stacked = torch.stack(probs_per_model, dim=0)
        
        # Apply weights element-wise and sum across the models
        weighted = stacked * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)