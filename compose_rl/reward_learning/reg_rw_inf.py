"""Inference model for loading and using the ClassifierRewardModel trained with ComposeRL."""

import os
import logging
from typing import Dict, Optional, Union, List, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class LlamaClassifierRewardModel:
    """
    A wrapper class for loading and using a trained ClassifierRewardModel for inference.
    This focuses only on the reward head component without LLM generation.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        """
        Initialize the reward model for inference.

        Args:
            model_path: Path to the trained model (local or S3)
            device: Device to load the model on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # Handle S3 paths by downloading to a temp location if needed
        if model_path.startswith("s3://"):
            try:
                import boto3
                from urllib.parse import urlparse

                parsed_url = urlparse(model_path)
                bucket_name = parsed_url.netloc
                key = parsed_url.path.lstrip("/")

                s3_client = boto3.client("s3")
                local_path = os.path.join("/tmp", os.path.basename(key))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                logger.info(f"Downloading model from {model_path} to {local_path}")
                s3_client.download_file(bucket_name, key, local_path)
                model_path = local_path
            except ImportError:
                raise ImportError(
                    "boto3 is required for loading from S3. Install with: pip install boto3"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model from S3: {e}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model with sequence classification head
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=self.torch_dtype, device_map=self.device
        )
        self.model.eval()

        # Check number of labels for the classifier
        self.num_labels = getattr(
            self.model.config, "num_labels", 5
        )  # Default to 5 for 0-4 range
        logger.info(f"Loaded classifier model with {self.num_labels} labels")

    def get_reward(
        self, prompts: Union[str, List[str]], responses: Union[str, List[str]], **kwargs
    ) -> torch.Tensor:
        """
        Get reward scores for prompt+response pairs.

        Args:
            prompts: Input prompt(s)
            responses: Response(s) to evaluate
            **kwargs: Additional arguments for tokenization

        Returns:
            Tensor of reward scores
        """
        # Ensure inputs are lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]

        assert len(prompts) == len(
            responses
        ), "Number of prompts and responses must match"

        # Prepare inputs (combining prompts and responses)
        inputs = []
        for prompt, response in zip(prompts, responses):
            inputs.append(f"{prompt}{response}")

        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            **kwargs,
        )

        # Move inputs to the correct device
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        # Get scores based on model output type
        if self.num_labels == 1:
            # Regression model (direct score)
            scores = outputs.logits.squeeze(-1)
        else:
            # Classification model
            # For a classifier with numerical ratings (e.g., 0-4 scale),
            # we can either take the expected value or the class with highest probability

            # Expected value approach (weighted average of class values)
            probs = F.softmax(outputs.logits, dim=1)
            class_values = torch.arange(
                0, self.num_labels, device=self.device, dtype=self.torch_dtype
            )
            scores = torch.sum(probs * class_values.unsqueeze(0), dim=1)

        return scores

    def batch_score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Score a pre-tokenized batch directly.

        Args:
            batch: Dictionary with 'input_ids' and 'attention_mask'

        Returns:
            Tensor of reward scores
        """
        # Ensure batch is on correct device
        batch = {
            k: v.to(self.device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**batch)

        # Get scores based on model output type
        if self.num_labels == 1:
            # Regression model (direct score)
            scores = outputs.logits.squeeze(-1)
        else:
            # Classification model with expected value
            probs = F.softmax(outputs.logits, dim=1)
            class_values = torch.arange(
                0, self.num_labels, device=self.device, dtype=self.torch_dtype
            )
            scores = torch.sum(probs * class_values.unsqueeze(0), dim=1)

        return scores


# Example usage
if __name__ == "__main__":
    model_path = (
        "s3://mybucket-jenny-test/rlhf-checkpoints/reg-rm/hf/huggingface/ba125/"
    )
    reward_model = LlamaClassifierRewardModel(model_path)

    # Example prompt-response pairs
    prompts = [
        "What is the capital of France?",
        "Explain how a transformer model works.",
    ]
    responses = [
        "The capital of France is Paris.",
        "A transformer model is a neural network architecture that uses self-attention mechanisms to process sequential data.",
    ]

    # Get reward scores
    scores = reward_model.get_reward(prompts, responses)
    print(f"Reward scores: {scores}")
