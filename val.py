import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from composer.trainer import Trainer  # Composer's Trainer to load .distcp checkpoints


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Path to Composer checkpoint directory (not individual files)
    model_checkpoint = "/tmp/reward_model/ep1-ba125/"
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"

    # ✅ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Free up memory
    torch.cuda.empty_cache()

    # ✅ Initialize the model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, num_labels=5
    ).half()
    model.to(device)
    model.eval()
    model = torch.compile(model)

    # ✅ Restore checkpoint correctly using Composer's Trainer
    trainer = Trainer(model=model)
    trainer.load_checkpoint(
        model_checkpoint
    )  # ✅ Loads all `.distcp` shards automatically

    # ✅ Load validation dataset (First 100 samples)
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

        # ✅ Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        # ✅ Calculate Accuracy
        if predicted_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"Accuracy on first {total} validation samples: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
