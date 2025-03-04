#!/usr/bin/env python3
"""
Classifier Reward Model Inference Script
"""

import os
import sys
import logging
import argparse
from typing import Dict, Optional, Union, List, Any, Mapping, MutableMapping

# Add llm-foundry to the Python path
sys.path.append("/llm-foundry")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Now import the necessary modules
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer

# Import the necessary components from compose_rl
# First, make sure compose_rl is in the path
compose_rl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if compose_rl_path not in sys.path:
    sys.path.append(compose_rl_path)

# Import the model components
try:
    from compose_rl.reward_learning.hf_utils import (
        RewardModelConfig,
        SequenceClassifierOutput,
    )
    from compose_rl.reward_learning.base_reward import RewardModel
    from compose_rl.reward_learning.model import ComposerHFClassifierRewardModel
except ImportError as e:
    logger.error(f"Failed to import from compose_rl: {e}")
    logger.info("Trying direct imports...")

    # Try direct imports as a fallback
    from hf_utils import RewardModelConfig, SequenceClassifierOutput
    from base_reward import RewardModel
    from model import ComposerHFClassifierRewardModel


class ClassifierRewardModel:
    """
    A wrapper class for loading and using a trained ComposerHFClassifierRewardModel for inference.
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

        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create a tokenizer wrapper that matches what ComposeRL expects
        tokenizer_wrapper = self._create_tokenizer_wrapper(self.tokenizer)

        # Initialize the model with the ComposeRL class
        logger.info(f"Loading model from {model_path}")
        self.model = ComposerHFClassifierRewardModel(
            pretrained_model_name_or_path=model_path,
            tokenizer=tokenizer_wrapper,
            use_train_metrics=False,
            loss_type="bce",
            return_last=True,
            return_lm_logits=False,
        )

        # Move model to device and set to evaluation mode
        if hasattr(self.model, "model"):
            self.model.model.to(device=self.device, dtype=self.torch_dtype)
            self.model.model.eval()
        else:
            self.model.to(device=self.device, dtype=self.torch_dtype)
            self.model.eval()

        # Get number of labels
        if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
            self.num_labels = getattr(self.model.model.config, "num_labels", 5)
        else:
            self.num_labels = 5  # Default to 5 for 0-4 range

        logger.info(f"Loaded classifier model with {self.num_labels} labels")

    def _create_tokenizer_wrapper(self, tokenizer):
        """
        Create a tokenizer wrapper that matches the expected interface for ComposeRL.
        This adapts the HF tokenizer to the interface expected by the ComposeRL framework.
        """

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                for attr in dir(tokenizer):
                    if not attr.startswith("_") and not hasattr(self, attr):
                        setattr(self, attr, getattr(tokenizer, attr))

            def encode_plus(self, *args, **kwargs):
                return self.tokenizer.encode_plus(*args, **kwargs)

            def batch_encode_plus(self, *args, **kwargs):
                return self.tokenizer.batch_encode_plus(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self.tokenizer(*args, **kwargs)

        return TokenizerWrapper(tokenizer)

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

        # Add is_inference flag for the model's forward method
        batch = {**tokenized_inputs, "is_inference": True}

        # Get model outputs using the model's forward method
        with torch.no_grad():
            scores = self.model.forward(batch)

        # Process scores based on model output type
        if isinstance(scores, dict) and "scores" in scores:
            scores = scores["scores"]

        if scores.dim() > 1 and scores.shape[1] > 1:
            # Classification model - compute expected value
            probs = F.softmax(scores, dim=1)
            class_values = torch.arange(
                0, scores.shape[1], device=self.device, dtype=scores.dtype
            )
            scores = torch.sum(probs * class_values.unsqueeze(0), dim=1)
        elif scores.dim() > 1:
            # Single label but in 2D format
            scores = scores.squeeze(1)

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

        # Add is_inference flag
        batch["is_inference"] = True

        # Get model outputs
        with torch.no_grad():
            scores = self.model.forward(batch)

        # Process scores based on model output type
        if isinstance(scores, dict) and "scores" in scores:
            scores = scores["scores"]

        if scores.dim() > 1 and scores.shape[1] > 1:
            # Classification model - compute expected value
            probs = F.softmax(scores, dim=1)
            class_values = torch.arange(
                0, scores.shape[1], device=self.device, dtype=scores.dtype
            )
            scores = torch.sum(probs * class_values.unsqueeze(0), dim=1)
        elif scores.dim() > 1:
            # Single label but in 2D format
            scores = scores.squeeze(1)

        return scores


def run_inference(model_path, prompts=None, responses=None):
    """
    Run inference with the classifier reward model
    """
    # Load the model
    reward_model = ClassifierRewardModel(model_path)

    # If no prompts/responses provided, use test examples
    if prompts is None or responses is None:
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

    # Print results
    for i, (prompt, response, score) in enumerate(zip(prompts, responses, scores)):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Score: {score.item():.2f}")

    return scores


def main():
    """
    Main entry point with command line argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a classifier reward model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="s3://mybucket-jenny-test/rlhf-checkpoints/reg-rm/hf/huggingface/ba125/",
        help="Path to the reward model (local or S3)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Optional single prompt to evaluate"
    )
    parser.add_argument(
        "--response",
        type=str,
        default=None,
        help="Optional single response to evaluate",
    )

    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else None
    responses = [args.response] if args.response else None

    run_inference(args.model_path, prompts, responses)


if __name__ == "__main__":
    main()
