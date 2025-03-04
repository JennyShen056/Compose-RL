import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import json
import boto3
from botocore.exceptions import ClientError
import logging
import sys

# Add Compose-RL to path if needed
compose_rl_path = "/Compose-RL"
if compose_rl_path not in sys.path:
    sys.path.append(compose_rl_path)

from compose_rl.reward_learning.modeling_hf import ComposerHFSequenceClassification
from compose_rl.reward_learning.model_methods import ClassifierRewardEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComposerHFClassifierRewardModel(ComposerHFSequenceClassification):
    """Implementation matching your trained model architecture"""

    def __init__(
        self,
        tokenizer,
        use_train_metrics=True,
        additional_train_metrics=None,
        additional_eval_metrics=None,
        loss_type="bce",
        return_lm_logits=False,
        return_last=True,
        **kwargs,
    ):
        self.loss_type = ClassifierRewardEnum(loss_type)
        self.return_lm_logits = return_lm_logits
        self.return_last = return_last

        config_overrides = {
            "num_labels": 5,  # For 0-4 range
            "return_logits": return_lm_logits,
        }

        if "config_overrides" in kwargs:
            config_overrides.update(kwargs.pop("config_overrides"))

        super().__init__(
            tokenizer=tokenizer,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            config_overrides=config_overrides,
            **kwargs,
        )


def download_from_s3(bucket_name, s3_prefix, local_dir):
    """Download model files from S3 to local directory"""
    s3_client = boto3.client("s3")

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    try:
        # List objects in the S3 bucket with the given prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                # Create local subdirectories if needed
                rel_path = key[len(s3_prefix) :].lstrip("/")
                local_file_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                logger.info(f"Downloading {key} to {local_file_path}")
                s3_client.download_file(bucket_name, key, local_file_path)
            logger.info(
                f"Downloaded all files from s3://{bucket_name}/{s3_prefix} to {local_dir}"
            )
        else:
            logger.warning(f"No objects found in s3://{bucket_name}/{s3_prefix}")

    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False

    return True


def format_conversation(example):
    """Format the conversation from the dataset into a prompt for the model"""
    conversation = example["text"]

    # Parse the conversation if it's a JSON string
    if isinstance(conversation, str):
        try:
            # Try to parse as JSON
            conversation = json.loads(conversation)
        except json.JSONDecodeError:
            # If not valid JSON, return as is
            return conversation

    # Create a formatted prompt
    formatted_text = ""
    for turn in conversation:
        role = turn.get("role", "").capitalize()
        content = turn.get("content", "")
        formatted_text += f"{role}: {content}\n\n"

    return formatted_text.strip()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model paths
    s3_bucket = "mybucket-jenny-test"
    s3_prefix = "rlhf-checkpoints/reg-rm/hf/huggingface/ba125/"
    local_model_dir = "./reward_model"

    # You can uncomment to download the model
    # download_from_s3(s3_bucket, s3_prefix, local_model_dir)

    # Use local path if available, otherwise try s3 path
    model_path = (
        local_model_dir
        if os.path.exists(local_model_dir)
        else f"s3://{s3_bucket}/{s3_prefix}"
    )

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=True
    )

    # Approach 1: Load directly with AutoModelForSequenceClassification
    logger.info("Loading model using AutoModelForSequenceClassification...")
    try:
        # Load model directly with AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, use_auth_token=True
        )
        direct_load_success = True
    except Exception as e:
        logger.error(f"Error loading with AutoModelForSequenceClassification: {e}")
        direct_load_success = False

    if not direct_load_success:
        # Approach 2: If Approach 1 fails, try with custom class
        logger.info(
            "Attempting to load with custom ComposerHFClassifierRewardModel class..."
        )
        try:
            # Load config
            config = AutoConfig.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                num_labels=5,
                trust_remote_code=True,
                use_auth_token=True,
            )

            # Initialize model with config first
            model = ComposerHFClassifierRewardModel(
                tokenizer=tokenizer,
                config=config,
                loss_type="bce",
                return_last=True,
                pretrained=True,
            )

            # Then load state dict
            state_dict = torch.load(f"{local_model_dir}/pytorch_model.bin")
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Error loading with custom class: {e}")
            logger.error("Both loading approaches failed. Exiting.")
            return

    model.to(device)
    model.eval()

    # Load the Jennny/Helpfulness dataset test split
    logger.info("Loading Helpfulness dataset...")
    dataset = load_dataset("Jennny/Helpfulness", split="test", use_auth_token=True)
    logger.info(f"Loaded {len(dataset)} test examples")

    # Prepare for inference
    predictions = []
    ground_truth = []

    # Process each example
    logger.info("Starting inference...")
    for i, example in enumerate(tqdm(dataset)):
        # Format conversation
        input_text = format_conversation(example)

        # Tokenize
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            # Handle different model types
            if direct_load_success:
                # Standard HF model
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                # Custom model
                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "is_inference": True,
                }
                outputs = model.forward(batch)
                logits = (
                    outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                )

            # Get predicted class (0-4)
            predicted_class = torch.argmax(logits, dim=-1).item()
            predictions.append(predicted_class)

            # Store ground truth
            ground_truth.append(example["labels"])

        # Print some examples
        if i < 5:  # Print first 5 examples
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Input: {input_text[:100]}...")
            logger.info(f"True label: {example['labels']}")
            logger.info(f"Predicted: {predicted_class}")

            # Print all logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            for cls_idx, prob in enumerate(probs):
                logger.info(f"Class {cls_idx}: {prob:.4f}")

    # Calculate metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    accuracy = accuracy_score(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)

    logger.info("\n===== Results =====")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(ground_truth, predictions))

    # Calculate error distribution
    error_dist = np.abs(predictions - ground_truth)

    logger.info("\nError Distribution:")
    for i in range(5):
        pct = (error_dist == i).mean() * 100
        logger.info(f"Error = {i}: {pct:.2f}%")

    # Save predictions to file
    output_file = "reward_model_predictions.csv"
    with open(output_file, "w") as f:
        f.write("true_label,predicted_label\n")
        for gt, pred in zip(ground_truth, predictions):
            f.write(f"{gt},{pred}\n")

    logger.info(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()
