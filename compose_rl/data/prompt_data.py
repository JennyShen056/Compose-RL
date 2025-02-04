# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import PreTrainedTokenizer
from typing import Any, List


def prompt_dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: List[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Collator for prompt data.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding text.
        max_seq_len (int): Maximum token sequence length.
        data (List[dict[str, torch.Tensor]]): List of prompt samples.

    Returns:
        dict: Collated batch for model input.
    """
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a PAD token.")

    input_ids, attention_masks = [], []

    for sample in data:
        text = sample["prompt"]

        # Truncate & pad
        if len(text) > max_seq_len:
            text = text[:max_seq_len]
        text = torch.cat(
            [text, torch.full((max_seq_len - len(text),), tokenizer.pad_token_id)]
        )

        attention_mask = (text != tokenizer.pad_token_id).long()

        input_ids.append(text)
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "text_attention_mask": torch.stack(attention_masks),
    }
