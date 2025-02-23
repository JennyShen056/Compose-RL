from typing import List, Optional, Any, Mapping, MutableMapping
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
import os

from datasets import load_dataset
from compose_rl.reward_learning.modeling_hf import ComposerHFSequenceClassification
from compose_rl.reward_learning.base_reward import RewardModel, Tokenizer
from compose_rl.reward_learning.model_methods import (
    PairwiseRewardEnum,
    pairwise_forward,
    pairwise_loss,
    ClassifierRewardEnum,
    classifier_loss,
    classifier_forward,
)
from compose_rl.reward_learning.hf_utils import SequenceClassifierOutput

# S3 configuration
S3_BUCKET = "mybucket-jenny-test"
S3_MODEL_PATH = "rlhf-checkpoints/reg-rm/hf/ba125"
LOCAL_MODEL_DIR = "/tmp/reward_model"

class ComposerHFClassifierRewardModel(ComposerHFSequenceClassification, RewardModel):
    def __init__(
        self,
        tokenizer: Tokenizer,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        loss_type: str = "bce",
        return_lm_logits: bool = False,
        return_last: bool = True,
        **kwargs: Any,
    ):
        self.loss_type = loss_type
        self.return_lm_logits = return_lm_logits
        self.return_last = return_last

        config_overrides = {
            "num_labels": 5,  # Reward range from 0-4
            "return_logits": return_lm_logits,
        }

        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            config_overrides=config_overrides,
            **kwargs,
        )

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        """Forward pass for reward inference."""
        return classifier_forward(
            model=self.model,
            tokenizer=self.tokenizer,
            batch=batch,
            return_last=self.return_last,
            return_lm_logits=self.return_lm_logits,
        )

    def eval_forward(self, batch: MutableMapping, outputs: Optional[SequenceClassifierOutput] = None) -> dict[str, torch.Tensor]:
        """Evaluation forward pass."""
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: SequenceClassifierOutput, batch: Mapping) -> dict[str, torch.Tensor]:
        """Compute loss for reward model training."""
        return classifier_loss(outputs, batch, self.loss_type)


class RewardModelHandler:
    """Handles loading and inference for the reward model."""

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def download_model_from_s3(self):
        """Download the model from S3 to local storage."""
        if not os.path.exists(LOCAL_MODEL_DIR):
            os.makedirs(LOCAL_MODEL_DIR)

        s3_client = boto3.client("s3")
        for obj in s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODEL_PATH)["Contents"]:
            s3_file_path = obj["Key"]
            local_file_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(s3_file_path))

            if not os.path.exists(local_file_path):
                s3_client.download_file(S3_BUCKET, s3_file_path, local_file_path)
                print(f"Downloaded {s3_file_path} to {local_file_path}")

    def load_model(self):
        """Load tokenizer and model from the local S3-downloaded directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)

        # Wrap in reward model interface
        self.model = ComposerHFClassifierRewardModel(tokenizer=self.tokenizer)

    def run_inference(self, input_text: str) -> float:
        """Run reward model inference on input text."""
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.forward(inputs)

        # Extract reward score
        reward_score = outputs["logits"].squeeze().item()
        return reward_score


# Load Hugging Face dataset
print("Loading dataset from Hugging Face...")
dataset = load_dataset("Jennny/Helpfulness")

# Select 5 samples from the validation split
val_samples = dataset["validation"].select(range(5))

# Extract the "text" column for inference
text_samples = val_samples["text"]

# Initialize and run reward model
if __name__ == "__main__":
    handler = RewardModelHandler()
    handler.download_model_from_s3()
    handler.load_model()

    print("\nRunning inference on 5 validation samples...\n")
    for i, text in enumerate(text_samples):
        reward = handler.run_inference(text)
        print(f"Sample {i+1} Reward Score: {reward}")