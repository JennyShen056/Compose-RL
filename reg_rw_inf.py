import os
import sys
import torch
import logging
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import boto3
from botocore.exceptions import ClientError

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

                # Skip directories
                if key.endswith("/"):
                    continue

                # Create local subdirectories if needed
                rel_path = key[len(s3_prefix) :].lstrip("/")
                local_file_path = os.path.join(local_dir, rel_path)

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                logger.info(f"Downloading {key} to {local_file_path}")
                s3_client.download_file(bucket_name, key, local_file_path)

        logger.info(f"Download complete")

        # Verify downloaded files
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

    # First make sure the model is downloaded
    download_success = download_from_s3(s3_bucket, s3_prefix, local_model_dir)

    if not download_success:
        logger.error("Failed to download model from S3. Exiting.")
        return

    # Check what files we have
    logger.info(f"Files in model directory: {os.listdir(local_model_dir)}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", token=os.environ["HF_TOKEN"]
    )

    # Simply try to load the model with AutoModelForSequenceClassification
    try:
        logger.info(f"Loading model from {local_model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_dir, trust_remote_code=True, token=os.environ["HF_TOKEN"]
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        # If direct loading fails, try initializing with base model and num_labels=5
        logger.warning(f"Error loading model directly: {e}")
        logger.info("Trying to load base model and then load classifier head...")

        try:
            # Initialize with base model
            model = AutoModelForSequenceClassification.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                num_labels=5,
                trust_remote_code=True,
                token=os.environ["HF_TOKEN"],
            )

            # Look for state dict files
            state_dict_files = [
                f
                for f in os.listdir(local_model_dir)
                if f.endswith(".bin") or f.endswith(".pt") or f.endswith(".safetensors")
            ]

            if state_dict_files:
                # Try to load state dict
                for file in state_dict_files:
                    try:
                        logger.info(f"Attempting to load state dict from {file}")
                        if file.endswith(".safetensors"):
                            from safetensors.torch import load_file

                            state_dict = load_file(os.path.join(local_model_dir, file))
                        else:
                            state_dict = torch.load(os.path.join(local_model_dir, file))

                        # Try to load
                        model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Successfully loaded state dict from {file}")
                        break
                    except Exception as e2:
                        logger.warning(f"Failed to load state dict from {file}: {e2}")
            else:
                logger.error("No state dict files found in model directory")
                return
        except Exception as e:
            logger.error(f"Error initializing base model: {e}")
            return

    model.to(device)
    model.eval()

    # Load the dataset
    logger.info("Loading Helpfulness dataset...")
    dataset = load_dataset("Jennny/Helpfulness", split="validation")
    dataset = dataset[:10]
    logger.info(f"Loaded {len(dataset)} test examples")

    # Let's examine the dataset structure first
    first_item = dataset[0]
    logger.info(f"Dataset item structure: {type(first_item)}")
    logger.info(
        f"Dataset item keys: {first_item.keys() if hasattr(first_item, 'keys') else 'N/A'}"
    )
    logger.info(f"First dataset item: {first_item}")

    # Prepare for inference
    predictions = []
    ground_truth = []

    # Process each example
    logger.info("Starting inference...")
    for i, example in enumerate(tqdm(dataset)):
        try:
            # Extract conversation and parse if needed
            if isinstance(example, dict) and "text" in example:
                conversation = example["text"]

                # If the conversation is a string, try to parse it as JSON
                if isinstance(conversation, str):
                    try:
                        conversation = json.loads(conversation)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse conversation as JSON: {conversation[:50]}..."
                        )
                        continue

                # Use tokenizer's chat template to format the conversation
                messages = conversation
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                # If the structure is not what we expect, log and skip
                logger.warning(f"Unexpected dataset structure: {example}")
                continue

            # Tokenize
            inputs = tokenizer(
                formatted_text, return_tensors="pt", truncation=True, max_length=4096
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Get predicted class (0-4)
                predicted_class = torch.argmax(logits, dim=-1).item()
                predictions.append(predicted_class)

                # Store ground truth
                if isinstance(example, dict) and "labels" in example:
                    ground_truth.append(example["labels"])
                else:
                    logger.warning(f"No labels found in example: {example}")
                    continue

            # Print some examples
            if i < 5:  # Print first 5 examples
                logger.info(f"\nExample {i+1}:")
                logger.info(f"Input: {formatted_text[:100]}...")
                logger.info(
                    f"True label: {example['labels'] if isinstance(example, dict) and 'labels' in example else 'N/A'}"
                )
                logger.info(f"Predicted: {predicted_class}")

                # Print all logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                for cls_idx, prob in enumerate(probs):
                    logger.info(f"Class {cls_idx}: {prob:.4f}")

        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue

    # Calculate metrics
    if len(predictions) > 0 and len(ground_truth) > 0:
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

        logger.info(f"Predictions saved to {output_file}")
    else:
        logger.error(
            "No valid predictions collected. Check the dataset format and processing."
        )


if __name__ == "__main__":
    main()
