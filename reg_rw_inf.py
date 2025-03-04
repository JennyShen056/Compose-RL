import os
import sys
import torch
import logging
import json
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import boto3
from botocore.exceptions import ClientError

# Import your custom reward model class.
# Make sure that the module path here matches where ComposerHFClassifierRewardModel is defined.
from compose_rl.reward_learning.modeling_hf import ComposerHFClassifierRewardModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_from_s3(bucket_name, s3_prefix, local_dir):
    """Download model files from S3 to local directory"""
    logger.info(
        f"Downloading model files from s3://{bucket_name}/{s3_prefix} to {local_dir}"
    )

    s3_client = boto3.client("s3")

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    try:
        # List objects in the S3 bucket with the given prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # skip directories
                rel_path = key[len(s3_prefix) :].lstrip("/")
                local_file_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                logger.info(f"Downloading {key} to {local_file_path}")
                s3_client.download_file(bucket_name, key, local_file_path)
        logger.info("Download complete")
        if os.path.exists(local_dir):
            files = os.listdir(local_dir)
            logger.info(f"Files in {local_dir}: {files}")
            if "config.json" in files:
                logger.info("Found config.json - model download appears successful")
            else:
                logger.warning("config.json not found - model might not be complete")
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model paths
    s3_bucket = "mybucket-jenny-test"
    s3_prefix = "rlhf-checkpoints/reg-rm/hf/huggingface/ba125/"
    local_model_dir = "/tmp/reward_model"

    # Download the model from S3
    download_success = download_from_s3(s3_bucket, s3_prefix, local_model_dir)
    if not download_success:
        logger.error("Failed to download model from S3. Exiting.")
        return

    logger.info(f"Files in model directory: {os.listdir(local_model_dir)}")

    # Load tokenizer (using your base model name and auth token)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=os.environ["HF_TOKEN"]
    )

    # Load configuration from the checkpoint so that your custom RewardModelConfig is used.
    try:
        logger.info("Loading model configuration from checkpoint...")
        config = AutoConfig.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            use_auth_token=os.environ["HF_TOKEN"],
        )
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    # Now load your custom classifier reward model using your custom class.
    try:
        logger.info(f"Loading custom classifier reward model from {local_model_dir}...")
        model = ComposerHFClassifierRewardModel.from_pretrained(
            local_model_dir,
            config=config,
            trust_remote_code=True,
            use_auth_token=os.environ["HF_TOKEN"],
        )
        logger.info("Custom reward model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading custom reward model: {e}")
        return

    model.to(device)
    model.eval()

    # Load the dataset (using the validation split)
    logger.info("Loading Helpfulness dataset...")
    dataset = load_dataset("Jennny/Helpfulness", split="validation")
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.info(f"Dataset features: {dataset.features}")
    logger.info(f"Dataset columns: {dataset.column_names}")

    # Print a sample to inspect format
    if len(dataset) > 0:
        sample_idx = 0
        logger.info(f"Sample item ({sample_idx}):")
        for col in dataset.column_names:
            logger.info(f"  {col}: {dataset[sample_idx][col]}")

    predictions = []
    ground_truth = []

    logger.info("Starting inference...")
    for i, example in enumerate(tqdm(dataset)):
        try:
            conversation_data = example["text"]
            # Parse JSON if necessary
            if isinstance(conversation_data, str):
                try:
                    conversation_messages = json.loads(conversation_data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse conversation JSON: {conversation_data[:100]}..."
                    )
                    continue
            else:
                conversation_messages = conversation_data

            # Format conversation using the tokenizer's chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation_messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(
                formatted_text, return_tensors="pt", truncation=True, max_length=4096
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                predictions.append(predicted_class)
                ground_truth.append(example["labels"])

            if i < 5:
                logger.info(f"\nExample {i+1}:")
                logger.info(f"Conversation: {conversation_messages}")
                logger.info(
                    f"Formatted input (first 200 chars): {formatted_text[:200]}..."
                )
                logger.info(f"True label: {example['labels']}")
                logger.info(f"Predicted: {predicted_class}")
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                logger.info("Class probabilities:")
                for cls_idx, prob in enumerate(probs):
                    logger.info(f"  Class {cls_idx}: {prob:.4f}")
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    if predictions and ground_truth:
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        accuracy = accuracy_score(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        logger.info("\n===== Results =====")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Mean Absolute Error: {mae:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(ground_truth, predictions))
        error_dist = np.abs(predictions - ground_truth)
        logger.info("\nError Distribution:")
        for i in range(5):
            pct = (error_dist == i).mean() * 100
            logger.info(f"Error = {i}: {pct:.2f}%")
        output_file = "reward_model_predictions.csv"
        with open(output_file, "w") as f:
            f.write("true_label,predicted_label\n")
            for gt, pred in zip(ground_truth, predictions):
                f.write(f"{gt},{pred}\n")
        logger.info(f"Predictions saved to {output_file}")
    else:
        logger.error("No valid predictions collected.")


if __name__ == "__main__":
    main()
