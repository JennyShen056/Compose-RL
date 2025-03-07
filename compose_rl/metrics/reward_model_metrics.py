# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

########## reward_model_metrics.py #############
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class PairwiseRewardClassificationAccuracy(Metric):
    """Pairwise reward classifcation accuracy.

    Computes the accuracy of a pairwise reward model, by the score of chosen
    being greater than the score of the rejected.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, **kwargs: Any):
        # State from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "correct",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: dict, output_logits: torch.Tensor):
        del output_logits
        bs, _ = batch["chosen_scores"].shape

        self.total += bs
        self.correct += (
            (batch["chosen_scores"] > batch["rejected_scores"]).sum().detach().cpu()
        )

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class ClassificationAccuracy(Metric):
    """Classification accuracy metric.

    Computes the accuracy of a classifier by comparing predictions from logits
    against ground truth labels. Handles both binary and multi-class
    classification.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        num_classes: int = 5,  # Changed from binary flag to num_classes
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes

        self.add_state(
            "correct",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: dict, output_logits: torch.Tensor):
        logits = batch["output_scores"]
        targets = batch["labels"].squeeze(-1)

        predictions = torch.argmax(logits, dim=1)
        self.correct += (predictions == targets).sum().detach().cpu()
        self.total += targets.shape[0]

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total

    # def __init__(
    #     self,
    #     binary: bool = True,
    #     threshold: float = 0.5,
    #     dist_sync_on_step: bool = False,
    #     **kwargs: Any,
    # ):
    #     """Initialize the metric.

    #     Args:
    #         binary: If True, treats as binary classification with sigmoid.
    #                If False, treats as multi-class with softmax.
    #         threshold: Decision threshold for binary classification
    #         dist_sync_on_step: Synchronize metric state across processes
    #     """
    #     super().__init__(dist_sync_on_step=dist_sync_on_step)
    #     self.binary = binary
    #     self.threshold = threshold

    #     self.add_state(
    #         "correct",
    #         default=torch.tensor(0.0),
    #         dist_reduce_fx="sum",
    #     )
    #     self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    # def update(self, batch: dict, output_logits: torch.Tensor):
    #     """Update state with predictions and targets.

    #     Args:
    #         batch: Dictionary containing 'output_scores' and 'labels'
    #         output_logits: `None`
    #     """
    #     del output_logits
    #     logits = batch["output_scores"]
    #     # TODO: this might break something, need to double check
    #     targets = batch["labels"].squeeze(-1)
    #     assert logits.shape[0] == targets.shape[0], "Batch sizes must match"

    #     if self.binary:
    #         # Binary classification
    #         probs = torch.sigmoid(logits.squeeze())
    #         predictions = (probs > self.threshold).long()
    #     else:
    #         # Multi-class classification
    #         probs = torch.softmax(logits, dim=1)
    #         predictions = torch.argmax(probs, dim=1)

    #     self.correct += (predictions == targets).sum().detach().cpu()
    #     self.total += targets.shape[0]

    # def compute(self):
    #     """Compute the accuracy."""
    #     assert isinstance(self.correct, Tensor)
    #     assert isinstance(self.total, Tensor)
    #     return self.correct / self.total
