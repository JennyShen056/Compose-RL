import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from compose_rl.reward_learning.model import ComposerHFClassifierRewardModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Manually specify the correct paths
    model_path = (
        "/tmp/reward_model/ep1-ba125/__0_0.distcp"  # <- Ensure this is correct!
    )
    tokenizer_path = "meta-llama/Llama-3.1-8B-Instruct"  # <- Use the original tokenizer

    # ✅ Manually load tokenizer instead of AutoTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ✅ Load Composer-trained model manually
    model = ComposerHFClassifierRewardModel.load_model(model_path, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    # 🔥 Load the validation dataset (First 100 samples)
    dataset = load_dataset("Jennny/Helpfulness", split="validation")
    subset = dataset.select(range(100))

    correct = 0
    total = 0

    for sample in subset:
        text = sample["text"]
        true_label = sample["labels"]

        # ✅ Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ Prepare batch for Composer model
        text_len = inputs["attention_mask"].sum(dim=1)
        batch = {
            "text": inputs["input_ids"],
            "text_attention_mask": inputs["attention_mask"],
            "text_len": text_len,
        }

        # ✅ Make prediction
        with torch.no_grad():
            outputs = model.forward(batch)
            predicted_label = torch.argmax(outputs["output_scores"], dim=1).item()

        # ✅ Calculate Accuracy
        if predicted_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"Accuracy on first {total} validation samples: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
