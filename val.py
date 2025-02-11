import os
import torch
import torch.distributed.checkpoint as dist_cp  # Updated import
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, Repository

# ====== CONFIGURATION ======
# Path where the sharded checkpoint is stored
distcp_checkpoint_path = "/tmp/reward_model/ep1-ba125"

# Hugging Face repository details (Change these)
HF_USERNAME = "Jennny"  # Change to your HF username
HF_MODEL_NAME = "help_reg_rew"  # Change to your desired model name

# Define the model path
hf_model_repo = f"{HF_USERNAME}/{HF_MODEL_NAME}"
model_save_path = "huggingface_model"

# The original model used for training (to load architecture correctly)
pretrained_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Adjust as needed

# ====== STEP 1: LOAD THE BASE MODEL ======
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    torch_dtype=torch.bfloat16,  # Ensure correct dtype
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

print("Base model loaded successfully.")

# ====== STEP 2: LOAD SHARDED CHECKPOINT WITH PROPER MAPPING ======
print("Loading sharded checkpoint from:", distcp_checkpoint_path)

# Initialize an empty state_dict
state_dict = {}

# Use FileSystemReader to load sharded checkpoint properly
reader = FileSystemReader(distcp_checkpoint_path)
dist_cp.load(state_dict=state_dict, storage_reader=reader)

# Ensure key mapping matches Hugging Face structure
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("model.", "")  # Remove prefix if needed
    new_state_dict[new_key] = value

# Load the model's state dict with strict=False to allow mismatches
model.load_state_dict(new_state_dict, strict=False)

print("Sharded checkpoint successfully loaded into model.")

# ====== STEP 3: SAVE THE MODEL IN HUGGING FACE FORMAT ======
# Create directory if not exists
os.makedirs(model_save_path, exist_ok=True)

# Save model and tokenizer in Hugging Face format
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"✅ Model saved successfully to {model_save_path}")

# ====== STEP 4: UPLOAD TO HUGGING FACE ======
print("Uploading model to Hugging Face...")

api = HfApi()
api.create_repo(repo_id=hf_model_repo, exist_ok=True)

# Clone the repository locally
repo = Repository(local_dir=model_save_path, clone_from=hf_model_repo)

# Add model card
readme_content = f"""# {HF_MODEL_NAME}

This is a fine-tuned LLaMA 3 reward model, trained using MosaicML and converted to Hugging Face format.
"""
with open(f"{model_save_path}/README.md", "w") as f:
    f.write(readme_content)

# Push to Hugging Face Hub
repo.git_add()
repo.git_commit("Upload converted model")
repo.git_push()

print(
    f"🎉 Model uploaded successfully! View it here: https://huggingface.co/{hf_model_repo}"
)
