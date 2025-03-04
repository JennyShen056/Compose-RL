import os
import json
import torch
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Union, Mapping, MutableMapping
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch import nn
from tqdm import tqdm
import logging
import tempfile
from urllib.parse import urlparse
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_s3_model(s3_uri: str, local_dir: str):
    """
    Download all files from the S3 URI (which represents a directory) to the local directory.
    """
    s3 = boto3.resource("s3")
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    bucket = s3.Bucket(bucket_name)
    logger.info(
        f"Downloading model files from bucket '{bucket_name}' with prefix '{prefix}' to {local_dir}"
    )
    for obj in bucket.objects.filter(Prefix=prefix):
        # Create local filename by removing the prefix from the object key.
        rel_path = os.path.relpath(obj.key, prefix)
        local_file_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        logger.info(f"Downloading s3://{bucket_name}/{obj.key} to {local_file_path}")
        bucket.download_file(obj.key, local_file_path)


class LlamaRewardHead(nn.Module):
    """Reward head for LLaMA models."""

    def __init__(self, hidden_size, n_labels=5, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class RewardModel(nn.Module):
    """Base class for reward models."""

    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def eval_forward(
        self, batch: MutableMapping, outputs: Optional[Any] = None
    ) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(batch)


class ClassifierRewardModel(RewardModel):
    """Custom classifier reward model for inference."""

    def __init__(
        self, model_path: str, base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    ):
        super().__init__()
        logger.info(f"Initializing ClassifierRewardModel with model from {model_path}")

        # If the model path is an S3 URI, download it locally.
        if model_path.startswith("s3://"):
            temp_dir = tempfile.mkdtemp(prefix="reward_model_")
            download_s3_model(model_path, temp_dir)
            local_model_path = temp_dir
            logger.info(f"Model downloaded to {local_model_path}")
        else:
            local_model_path = model_path

        # Load the configuration first (with trust_remote_code=True)
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
        # Workaround: update config.model_type to a recognized value (e.g., "llama")
        config.model_type = "llama"

        # Then load the model with the updated config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        # Set additional parameters
        self.return_lm_logits = False
        self.return_last = True

        # Move model to device and set evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model type: {type(self.model)}")
        logger.info(f"Number of labels: {self.model.config.num_labels}")

    def forward(
        self, batch: MutableMapping
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for inference."""
        is_inference = batch.get("is_inference", True)
        if is_inference:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                # Use logits (or scores) from the output
                if hasattr(outputs, "logits"):
                    scores = outputs.logits
                else:
                    scores = outputs.scores
                if scores.shape[-1] > 1:
                    # Multi-class: return raw logits
                    return {"scores": scores}
                else:
                    # Binary/regression: squeeze last dimension
                    return {"scores": scores.squeeze(-1)}
        else:
            raise NotImplementedError(
                "Training forward pass not implemented for inference-only model"
            )


def parse_conversation_json(text: str) -> List[Dict[str, str]]:
    """
    Parse a conversation JSON string to a list of message dictionaries.
    """
    try:
        if isinstance(text, str):
            if text.startswith("[") and text.endswith("]"):
                return json.loads(text)
            else:
                text = text.replace("'", '"')
                return json.loads(text)
        else:
            return text
    except json.JSONDecodeError:
        logger.error(f"Failed to parse conversation: {text}")
        return []


def run_inference(
    model_path: str,
    dataset_name: str = "Jenny/Helpfulness",
    max_samples: int = 100,
    batch_size: int = 8,
    max_length: int = 2048,
    output_file: str = "reward_scores.jsonl",
):
    """
    Run inference on the specified dataset using the classifier reward model.
    """
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", padding_side="right", trust_remote_code=True
    )

    logger.info(f"Initializing model from {model_path}...")
    model = ClassifierRewardModel(model_path)

    logger.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    if "text" in dataset["train"].column_names:
        texts = dataset["train"]["text"]
        labels = (
            dataset["train"]["labels"]
            if "labels" in dataset["train"].column_names
            else None
        )
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a 'text' column")

    texts = texts[:max_samples]
    if labels is not None:
        labels = labels[:max_samples]

    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size] if labels is not None else None

        processed_texts = []
        for text in batch_texts:
            try:
                conversation = parse_conversation_json(text)
                # Use apply_chat_template if available; otherwise join conversation messages.
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_text = tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )
                else:
                    formatted_text = " ".join(
                        [msg.get("content", "") for msg in conversation]
                    )
                processed_texts.append(formatted_text)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                processed_texts.append("Hello")

        inputs = tokenizer(
            processed_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        inputs["is_inference"] = True

        with torch.no_grad():
            outputs = model.forward(inputs)
            batch_scores = outputs["scores"].cpu()

            if batch_scores.dim() > 1 and batch_scores.shape[1] > 1:
                predicted_classes = torch.argmax(batch_scores, dim=1).tolist()
                score_weights = torch.tensor(
                    [0, 1, 2, 3, 4], device=batch_scores.device
                ).float()
                weighted_scores = torch.softmax(batch_scores, dim=1) @ score_weights
                weighted_scores = weighted_scores.tolist()
            else:
                predicted_classes = None
                weighted_scores = batch_scores.squeeze().tolist()
                if not isinstance(weighted_scores, list):
                    weighted_scores = [weighted_scores]

        for j, (text, score) in enumerate(zip(batch_texts, weighted_scores)):
            result = {
                "text": text,
                "reward_score": score,
            }
            if predicted_classes is not None:
                result["predicted_class"] = predicted_classes[j]
            if batch_labels is not None:
                result["true_label"] = int(batch_labels[j])
            results.append(result)

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    scores = [r["reward_score"] for r in results]
    logger.info(f"\nInference completed on {len(scores)} samples")
    logger.info(f"Average score: {np.mean(scores):.4f}")
    logger.info(f"Min score: {min(scores):.4f}")
    logger.info(f"Max score: {max(scores):.4f}")

    if labels is not None:
        true_labels = [r["true_label"] for r in results]
        correlation = np.corrcoef(scores, true_labels)[0, 1]
        logger.info(f"Correlation with true labels: {correlation:.4f}")

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a classifier reward model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="s3://mybucket-jenny-test/rlhf-checkpoints/reg-rm/hf/huggingface/ba125/",
        help="Path to the reward model checkpoint (local path or S3 URI)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Jenny/Helpfulness",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="reward_scores.jsonl",
        help="File to save results to",
    )

    args = parser.parse_args()
    run_inference(
        model_path=args.model_path,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
