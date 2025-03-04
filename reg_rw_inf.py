import os
import sys
import torch
import logging
import json
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import boto3
from botocore.exceptions import ClientError
from safetensors import safe_open

# Import your custom reward model class
from compose_rl.reward_learning.model import ComposerHFClassifierRewardModel

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


def find_classifier_weights_in_safetensors(model_dir):
    """Find the safetensors files that contain classifier/score weights"""
    classifier_weights = {}

    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(model_dir, filename)
            try:
                with safe_open(file_path, framework="pt") as f:
                    tensor_names = f.keys()
                    for name in tensor_names:
                        if (
                            "score" in name
                            or "classifier" in name
                            or "out_proj" in name
                            or "lm_head" in name
                        ):
                            logger.info(f"Found relevant weight in {filename}: {name}")
                            tensor = f.get_tensor(name)
                            classifier_weights[name] = tensor
            except Exception as e:
                logger.warning(f"Error opening {filename}: {e}")

    if classifier_weights:
        logger.info(f"Found {len(classifier_weights)} classifier-related weights")
    else:
        logger.warning("No classifier weights found in safetensors files")

    return classifier_weights


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model paths
    s3_bucket = "mybucket-jenny-test"
    s3_prefix = "rlhf-checkpoints/reg-rm/hf/huggingface/ba125/"
    local_model_dir = "/tmp/reward_model"

    # Download the model from S3 (uncomment if needed)
    # download_success = download_from_s3(s3_bucket, s3_prefix, local_model_dir)
    # if not download_success:
    #     logger.error("Failed to download model from S3. Exiting.")
    #     return

    logger.info(f"Files in model directory: {os.listdir(local_model_dir)}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", token=os.environ["HF_TOKEN"]
    )

    # APPROACH 1: Use standard HuggingFace model and load weights
    logger.info("Initializing base classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_labels=5,  # 0-4 range for helpfulness
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"],
    )

    # Find and load classifier weights
    logger.info("Searching for classifier weights...")
    classifier_weights = find_classifier_weights_in_safetensors(local_model_dir)

    if classifier_weights:
        logger.info("Updating model with classifier weights...")
        state_dict = model.state_dict()

        # Map weights from checkpoint to model
        for name, tensor in classifier_weights.items():
            # Try to find matching key
            short_name = name.split(".")[-1]
            matching_keys = [k for k in state_dict.keys() if short_name in k]

            if matching_keys:
                match_key = matching_keys[0]
                if state_dict[match_key].shape == tensor.shape:
                    logger.info(f"Updating weight: {match_key}")
                    state_dict[match_key] = tensor
                else:
                    logger.warning(
                        f"Shape mismatch for {match_key}: {state_dict[match_key].shape} vs {tensor.shape}"
                    )
            else:
                if name in state_dict:
                    logger.info(f"Updating weight directly: {name}")
                    state_dict[name] = tensor
                else:
                    logger.warning(f"No matching key found for {name}")

        # Load updated weights
        model.load_state_dict(state_dict)
        logger.info("Model weights updated with classifier weights")

    model.to(device)
    model.eval()

    # Load the dataset
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
    class_distribution = {i: 0 for i in range(5)}

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
                class_distribution[predicted_class] += 1

            if i < 5 or i % 50 == 0:  # Show first 5 examples and then every 50th
                logger.info(f"\nExample {i+1}:")
                logger.info(f"True label: {example['labels']}")
                logger.info(f"Predicted: {predicted_class}")
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                logger.info("Class probabilities:")
                for cls_idx, prob in enumerate(probs):
                    logger.info(f"  Class {cls_idx}: {prob:.4f}")

                # Show distribution so far
                if i % 50 == 0:
                    logger.info(f"Class distribution so far: {class_distribution}")
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
        logger.info(f"Final class distribution: {class_distribution}")
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
