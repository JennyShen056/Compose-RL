import os
import json
import torch
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Union, Mapping, MutableMapping
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForCausalLM
from transformers import LlamaConfig, LlamaTokenizer, LlamaForSequenceClassification
from torch import nn
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """Base class for reward models to match the interface of ComposerHFClassifierRewardModel."""
    
    def forward(self, batch: MutableMapping) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def eval_forward(self, batch: MutableMapping, outputs: Optional[Any] = None) -> dict[str, torch.Tensor]:
        return outputs if outputs is not None else self.forward(batch)

class ClassifierRewardModel(RewardModel):
    """Custom implementation of a classifier reward model for inference."""
    
    def __init__(self, model_path: str, base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__()
        logger.info(f"Initializing ClassifierRewardModel with model from {model_path}")
        
        # Load the model directly - for inference we can use the HuggingFace model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Set model parameters
        self.return_lm_logits = False
        self.return_last = True
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model type: {type(self.model)}")
        logger.info(f"Number of labels: {self.model.config.num_labels}")

    def forward(self, batch: MutableMapping) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for inference."""
        # Check if this is an inference call
        is_inference = batch.get("is_inference", True)
        
        if is_inference:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Handle different output formats
                if hasattr(outputs, "logits"):
                    scores = outputs.logits
                else:
                    scores = outputs.scores
                
                # For classifier models, return the predicted class or raw logits
                if scores.shape[-1] > 1:  # Multi-class classification
                    # Return raw logits
                    return {"scores": scores}
                else:  # Binary classification or regression
                    return {"scores": scores.squeeze(-1)}
        else:
            # This path would include the full classifier_forward implementation
            # For simplicity in inference, we're only implementing the inference path
            raise NotImplementedError("Training forward pass not implemented for inference-only model")

def parse_conversation_json(text: str) -> List[Dict[str, str]]:
    """
    Parse conversation JSON string to list of message dictionaries.
    
    Args:
        text: JSON string of conversation
        
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    try:
        # Handle potential string escaping issues
        if isinstance(text, str):
            if text.startswith("[") and text.endswith("]"):
                return json.loads(text)
            else:
                # Try to fix common JSON issues
                text = text.replace("'", "\"")
                return json.loads(text)
        else:
            return text  # Already parsed
    except json.JSONDecodeError:
        logger.error(f"Failed to parse conversation: {text}")
        return []

def run_inference(
    model_path: str,
    dataset_name: str = "Jenny/Helpfulness",
    max_samples: int = 100,
    batch_size: int = 8,
    max_length: int = 2048,
    output_file: str = "reward_scores.jsonl"
):
    """
    Run inference on the specified dataset using the classifier reward model.
    
    Args:
        model_path: Path to the reward model checkpoint
        dataset_name: Name of the dataset to use
        max_samples: Maximum number of samples to process
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        output_file: File to save results to
    """
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        padding_side="right"
    )
    
    # Initialize model
    logger.info(f"Initializing model from {model_path}...")
    model = ClassifierRewardModel(model_path)
    
    # Load dataset
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    # Get text column data
    if "text" in dataset["train"].column_names:
        texts = dataset["train"]["text"]
        # Get labels if available
        labels = dataset["train"]["labels"] if "labels" in dataset["train"].column_names else None
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a 'text' column")
    
    # Limit to max_samples
    texts = texts[:max_samples]
    if labels is not None:
        labels = labels[:max_samples]
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size] if labels is not None else None
        
        # Process and format each conversation
        processed_texts = []
        for text in batch_texts:
            try:
                # Parse the conversation JSON
                conversation = parse_conversation_json(text)
                
                # Use apply_chat_template for proper formatting
                formatted_text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                processed_texts.append(formatted_text)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                # Create a minimal valid conversation for fallback
                fallback_conversation = [{"role": "user", "content": "Hello"}]
                fallback_text = tokenizer.apply_chat_template(
                    fallback_conversation, 
                    tokenize=False,
                    add_generation_prompt=False
                )
                processed_texts.append(fallback_text)
        
        # Tokenize
        inputs = tokenizer(
            processed_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        
        # Add is_inference flag
        inputs["is_inference"] = True
        
        # Run inference
        with torch.no_grad():
            outputs = model.forward(inputs)
            batch_scores = outputs["scores"].cpu()
            
            # Handle multi-class case - get prediction or scores
            if batch_scores.dim() > 1 and batch_scores.shape[1] > 1:
                # For 5-class (0-4) classification, we can either:
                # 1. Get the predicted class
                predicted_classes = torch.argmax(batch_scores, dim=1).tolist()
                # 2. Or get the expected value (weighted average)
                score_weights = torch.tensor([0, 1, 2, 3, 4], device=batch_scores.device).float()
                weighted_scores = torch.softmax(batch_scores, dim=1) @ score_weights
                weighted_scores = weighted_scores.tolist()
            else:
                # Binary classification or regression
                predicted_classes = None
                weighted_scores = batch_scores.squeeze().tolist()
                
                # Convert to list if only one item
                if not isinstance(weighted_scores, list):
                    weighted_scores = [weighted_scores]
        
        # Store results
        for j, (text, score) in enumerate(zip(batch_texts, weighted_scores)):
            result = {
                "text": text,
                "reward_score": score,
            }
            
            # Add predicted class if available
            if predicted_classes is not None:
                result["predicted_class"] = predicted_classes[j]
                
            # Add true label if available
            if batch_labels is not None:
                result["true_label"] = int(batch_labels[j])
                
            results.append(result)
    
    # Save results
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary statistics
    scores = [r["reward_score"] for r in results]
    logger.info(f"\nInference completed on {len(scores)} samples")
    logger.info(f"Average score: {np.mean(scores):.4f}")
    logger.info(f"Min score: {min(scores):.4f}")
    logger.info(f"Max score: {max(scores):.4f}")
    
    # If we have true labels, calculate correlation
    if labels is not None:
        true_labels = [r["true_label"] for r in results]
        correlation = np.corrcoef(scores, true_labels)[0, 1]
        logger.info(f"Correlation with true labels: {correlation:.4f}")
    
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with a classifier reward model")
    parser.add_argument("--model_path", type=str, default="s3://mybucket-jenny-test/rlhf-checkpoints/reg-rm/hf/huggingface/ba125/",
                        help="Path to the reward model checkpoint")
    parser.add_argument("--dataset", type=str, default="Jenny/Helpfulness",
                        help="HuggingFace dataset name")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--output_file", type=str, default="reward_scores.jsonl",
                        help="File to save results to")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
