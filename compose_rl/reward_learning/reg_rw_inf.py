#!/usr/bin/env python3
"""
Classifier Reward Model Inference Script with improved S3 handling
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


def find_model_in_s3(s3_path):
    """
    Explore S3 path to find the actual model path
    """
    try:
        import boto3
        from urllib.parse import urlparse

        # Parse the S3 URI
        parsed_url = urlparse(s3_path)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip("/")

        # Initialize S3 client
        s3_client = boto3.client("s3")
        logger.info(f"Exploring S3 bucket: {bucket_name}, prefix: {prefix}")

        # If prefix ends with a slash, it's likely a directory
        if prefix.endswith("/"):
            prefix = prefix.rstrip("/")

        # Paths to explore
        paths_to_check = [
            prefix,  # Original path
            f"{prefix}/",  # Original path as directory
            f"{prefix}ba125/",  # Specific subfolder mentioned in error
            "rlhf-checkpoints/reg-rm/hf/latest/",  # Common pattern
            "rlhf-checkpoints/reg-rm/latest/",  # Another common pattern
        ]

        # Check for HF model files in each path
        for path in paths_to_check:
            logger.info(f"Checking path: {path}")

            # Check for model files
            model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
            for file in model_files:
                try:
                    file_key = f"{path}{file}"
                    s3_client.head_object(Bucket=bucket_name, Key=file_key)
                    logger.info(f"Found model file: {file} at {path}")
                    return f"s3://{bucket_name}/{path}"
                except Exception as e:
                    # File not found at this path, try next file or path
                    pass

            # Check if the path exists as a directory containing files
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=path, Delimiter="/", MaxKeys=10
                )

                if "Contents" in response:
                    logger.info(f"Found directory at: {path}")
                    # Look for model files
                    for obj in response["Contents"]:
                        file_name = os.path.basename(obj["Key"])
                        if file_name in model_files:
                            logger.info(f"Found model file: {file_name} at {path}")
                            return f"s3://{bucket_name}/{path}"
            except Exception as e:
                logger.error(f"Error listing objects at {path}: {e}")

        logger.warning(f"Could not find model files in any of the expected paths")
        return s3_path  # Return original path if nothing found

    except Exception as e:
        logger.error(f"Error exploring S3: {e}")
        return s3_path


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

        # If path is S3, try to find the actual model files
        if model_path.startswith("s3://"):
            try:
                # First, try to explore S3 to find the model
                model_path = find_model_in_s3(model_path)
                logger.info(f"Using model path: {model_path}")

                # Now try to load directly from the path
                logger.info("Attempting to load directly from S3 path")
            except Exception as e:
                logger.warning(f"Error exploring S3: {e}")

        # Try to load the model with various approaches
        # First, try loading the tokenizer
        try:
            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
            logger.info(
                f"Successfully loaded tokenizer: {self.tokenizer.__class__.__name__}"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Try a default tokenizer as fallback
            try:
                logger.info(
                    "Falling back to meta-llama/Meta-Llama-3.1-8B-Instruct tokenizer"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    trust_remote_code=True,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                )
            except Exception as e2:
                logger.error(f"Failed to load fallback tokenizer: {e2}")
                raise RuntimeError("Could not load any tokenizer")

        # Create a tokenizer wrapper that matches what ComposeRL expects
        tokenizer_wrapper = self._create_tokenizer_wrapper(self.tokenizer)

        # Initialize the model with the ComposeRL class
        try:
            logger.info(f"Loading model from {model_path}")
            # Add extra parameters to handle various loading scenarios
            self.model = ComposerHFClassifierRewardModel(
                pretrained_model_name_or_path=model_path,
                tokenizer=tokenizer_wrapper,
                use_train_metrics=False,
                loss_type="bce",
                return_last=True,
                return_lm_logits=False,
                use_auth_token=os.environ.get("HF_TOKEN"),  # Use HF token if available
                trust_remote_code=True,
            )
            logger.info("Successfully loaded model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

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
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Alternative local path to model if S3 fails",
    )
    parser.add_argument(
        "--aws_access_key", type=str, default=None, help="AWS access key for S3 access"
    )
    parser.add_argument(
        "--aws_secret_key", type=str, default=None, help="AWS secret key for S3 access"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for accessing models",
    )

    args = parser.parse_args()

    # Set AWS credentials if provided
    if args.aws_access_key and args.aws_secret_key:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_key
        logger.info("Using provided AWS credentials")

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Use local path as fallback if S3 fails
    model_path = args.model_path
    try:
        prompts = [args.prompt] if args.prompt else None
        responses = [args.response] if args.response else None

        run_inference(model_path, prompts, responses)
    except Exception as e:
        logger.error(f"Error using primary model path: {e}")
        if args.local_model_path:
            logger.info(f"Falling back to local model path: {args.local_model_path}")
            run_inference(args.local_model_path, prompts, responses)
        else:
            logger.error("No fallback path provided. Exiting.")
            raise


if __name__ == "__main__":
    main()
