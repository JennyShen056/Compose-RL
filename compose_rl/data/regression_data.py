# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a regression reward dataset and dataloader for training."""

import logging
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

log = logging.getLogger(__name__)


def regression_dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Collator for regression reward model data.

    Args:
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (list[dict[str, torch.Tensor]]): The regression data to collate.
    """
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a PAD token.")

    input_ids = []
    attention_masks = []
    labels = []

    for sample in data:
        text = sample["text"]
        label = sample["label"]

        if len(text) > max_seq_len:
            text = text[:max_seq_len]  # Truncate to max_seq_len
        text = torch.cat(
            [text, torch.full((max_seq_len - len(text),), tokenizer.pad_token_id)]
        )

        attention_mask = (text != tokenizer.pad_token_id).long()

        input_ids.append(text)
        attention_masks.append(attention_mask)
        labels.append(label)

    return {
        "input_ids": torch.stack(input_ids),
        "text_attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


class RegressionStreamingDataset(StreamingDataset):
    """Dataloader for streaming in regression reward model data."""

    def __init__(self, max_seq_len: int, **kwargs: dict[str, Any]):
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from StreamingDataset at a given index.

        Args:
            idx (int): the index where we fetch the data in the StreamingDataset.
        """
        sample = super().__getitem__(idx)
        text = torch.from_numpy(np.frombuffer(sample["text"], dtype=np.int64))
        label = torch.tensor(sample["label"], dtype=torch.float32)

        return {
            "text": text[: self.max_seq_len],  # Truncate if necessary
            "label": label.unsqueeze(0),
        }
