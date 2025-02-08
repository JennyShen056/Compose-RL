#!/usr/bin/env python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Import the custom classifier reward model.
# (If your model was saved with a custom class, ensure this import works correctly.)
from Compose_RL.compose_rl.reward_learning.model import ComposerHFClassifierRewardModel


def main():
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load the model and tokenizer ---
    model_path = "/tmp/reward_model"
    # Load tokenizer from the saved folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load the saved model; note that we pass the tokenizer as required by your custom model.
    model = ComposerHFClassifierRewardModel.from_pretrained(
        model_path, tokenizer=tokenizer
    )
    model.to(device)
    model.eval()

    # --- Load the dataset (validation split, first 100 samples) ---
    dataset = load_dataset("Jennny/Helpfulness", split="validation")
    # Select the first 100 samples
    subset = dataset.select(range(100))

    correct = 0
    total = 0

    for sample in subset:
        # Get the raw text and true label from the sample
        text = sample["text"]
        true_label = sample["labels"]

        # --- Tokenize the input text ---
        # Here we use a maximum length of 512; adjust if necessary.
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # --- Prepare the expected keys for the model ---
        # Your classifier_forward expects: "text", "text_attention_mask", and "text_len".
        text_len = inputs["attention_mask"].sum(dim=1)  # Number of non-padded tokens
        batch = {
            "text": inputs["input_ids"],
            "text_attention_mask": inputs["attention_mask"],
            "text_len": text_len,
        }

        # --- Make prediction ---
        with torch.no_grad():
            outputs = model.forward(batch)
            # The classifier forward returns a dict with key "output_scores"
            # Predict the class as the argmax over the output scores.
            predicted_label = torch.argmax(outputs["output_scores"], dim=1).item()

        # --- Compare prediction with the true label ---
        if predicted_label == true_label:
            correct += 1
        total += 1

    # --- Compute and print the accuracy ---
    accuracy = correct / total * 100
    print(f"Accuracy on first {total} validation samples: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
