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
from safetensors import safe_open

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

    # Set the HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        return

    # Model paths
    s3_bucket = "mybucket-jenny-test"
    s3_prefix = "rlhf-checkpoints/reg-rm/hf/huggingface/ba125/"
    local_model_dir = "/tmp/reward_model"

    # Uncomment if you need to download the model again
    # download_success = download_from_s3(s3_bucket, s3_prefix, local_model_dir)
    # if not download_success:
    #     logger.error("Failed to download model from S3. Exiting.")
    #     return

    logger.info(f"Files in model directory: {os.listdir(local_model_dir)}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", token=hf_token
    )

    # Initialize the model with 5 classes
    logger.info("Initializing base model with 5 classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_labels=5,
        trust_remote_code=True,
        token=hf_token,
    )

    # IMPORTANT: Only load the classification head weights (score.weight)
    logger.info("Loading classification head weights only...")

    # Initialize score.weight properly
    score_head_loaded = False

    # Examine each safetensors file to find the score.weight
    for filename in sorted(os.listdir(local_model_dir)):
        if filename.endswith(".safetensors"):
            try:
                logger.info(f"Examining {filename} for score.weight...")
                with safe_open(
                    os.path.join(local_model_dir, filename), framework="pt"
                ) as f:
                    if "score.weight" in f.keys():
                        logger.info(f"Found score.weight in {filename}")
                        # Get the tensor and print its shape
                        score_weight = f.get_tensor("score.weight")
                        logger.info(f"score.weight shape: {score_weight.shape}")

                        # Check if shape matches
                        if model.score.weight.shape == score_weight.shape:
                            logger.info(f"Shapes match! Loading score.weight")
                            model.score.weight.data.copy_(score_weight)
                            score_head_loaded = True
                            break
                        else:
                            logger.warning(
                                f"Shape mismatch: model expects {model.score.weight.shape}, found {score_weight.shape}"
                            )
            except Exception as e:
                logger.error(f"Error examining {filename}: {e}")

    # if not score_head_loaded:
    #     logger.warning("Failed to find compatible score.weight in model files")
    #     # As a fallback, let's create a simple linear projection from hidden size to num_labels
    #     logger.info("Creating a simple classifier head...")
    #     # Get the model's hidden size
    #     hidden_size = model.config.hidden_size
    #     # Initialize with Xavier initialization
    #     nn_init = torch.nn.init.xavier_uniform_
    #     model.score.weight.data = nn_init(torch.zeros(5, hidden_size))

    model.to(device)
    model.eval()

    # Load the dataset
    logger.info("Loading Helpfulness dataset...")
    dataset = load_dataset("Jennny/Helpfulness", split="validation")
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.info(f"Dataset features: {dataset.features}")
    logger.info(f"Dataset columns: {dataset.column_names}")

    # Get label distribution in dataset
    label_counts = {}
    for ex in dataset:
        label = ex["labels"]
        label_counts[label] = label_counts.get(label, 0) + 1

    logger.info(f"Label distribution in dataset: {label_counts}")

    # Print a sample to inspect format
    if len(dataset) > 0:
        sample_idx = 0
        logger.info(f"Sample item ({sample_idx}):")
        for col in dataset.column_names:
            logger.info(f"  {col}: {dataset[sample_idx][col]}")

    # Prepare for inference
    predictions = []
    ground_truth = []
    class_distribution = {i: 0 for i in range(5)}

    # Process in smaller batches to see progress
    logger.info("Starting inference...")

    # Create a smaller sample for debugging if needed
    # dataset = dataset.select(range(min(100, len(dataset))))

    for i, example in enumerate(tqdm(dataset)):
        try:
            # Get the text and labels
            conversation_data = example["text"]
            true_label = example["labels"]

            # Parse JSON if needed
            if isinstance(conversation_data, str):
                try:
                    conversation_messages = json.loads(conversation_data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse conversation as JSON: {conversation_data[:100]}..."
                    )
                    continue
            else:
                conversation_messages = conversation_data

            # Format using chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation_messages, tokenize=False, add_generation_prompt=False
            )

            # Tokenize
            inputs = tokenizer(
                formatted_text, return_tensors="pt", truncation=True, max_length=4096
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Get predicted class
                predicted_class = torch.argmax(logits, dim=-1).item()

                # Store results
                predictions.append(predicted_class)
                ground_truth.append(true_label)
                class_distribution[predicted_class] += 1

            # Show examples
            if i < 5 or i % 100 == 0:
                logger.info(f"\nExample {i+1}:")
                logger.info(f"True label: {true_label}")
                logger.info(f"Predicted: {predicted_class}")

                # Calculate probabilities
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                logger.info("Class probabilities:")
                for cls_idx, prob in enumerate(probs):
                    logger.info(f"  Class {cls_idx}: {prob:.4f}")

                # Show running statistics
                if i > 0 and i % 100 == 0:
                    logger.info(f"Class distribution so far: {class_distribution}")

                    # Calculate running metrics
                    running_preds = np.array(predictions)
                    running_truth = np.array(ground_truth)
                    running_acc = accuracy_score(running_truth, running_preds)
                    running_mae = mean_absolute_error(running_truth, running_preds)
                    logger.info(f"Running accuracy: {running_acc:.4f}")
                    logger.info(f"Running MAE: {running_mae:.4f}")

        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Calculate final metrics
    logger.info("\nCalculating final metrics...")
    if len(predictions) > 0 and len(ground_truth) > 0:
        try:
            predictions_array = np.array(predictions)
            ground_truth_array = np.array(ground_truth)

            logger.info(f"Number of predictions: {len(predictions_array)}")
            logger.info(f"Number of ground truth labels: {len(ground_truth_array)}")

            # Basic metrics
            accuracy = accuracy_score(ground_truth_array, predictions_array)
            mae = mean_absolute_error(ground_truth_array, predictions_array)

            logger.info("\n===== Results =====")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Mean Absolute Error: {mae:.4f}")
            logger.info(f"Final class distribution: {class_distribution}")

            # Detailed classification report
            logger.info("\nClassification Report:")
            class_report = classification_report(ground_truth_array, predictions_array)
            logger.info(f"\n{class_report}")

            # Error distribution
            error_dist = np.abs(predictions_array - ground_truth_array)
            logger.info("\nError Distribution:")
            for i in range(5):
                count = (error_dist == i).sum()
                pct = (error_dist == i).mean() * 100
                logger.info(f"Error = {i}: {count} examples ({pct:.2f}%)")

            # Save predictions to CSV
            output_file = "reward_model_predictions.csv"
            with open(output_file, "w") as f:
                f.write("true_label,predicted_label\n")
                for gt, pred in zip(ground_truth, predictions):
                    f.write(f"{gt},{pred}\n")

            logger.info(f"\nPredictions saved to {output_file}")

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            import traceback

            logger.error(traceback.format_exc())

    else:
        logger.error("No valid predictions collected.")


if __name__ == "__main__":
    main()
