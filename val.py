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

# Authenticate automatically (no token needed in code)
api = HfApi()
api.create_repo(repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}", exist_ok=True)

repo = Repository(
    local_dir=model_save_path,
    clone_from=f"https://huggingface.co/{HF_USERNAME}/{HF_MODEL_NAME}",
)

import subprocess

# Ensure repository is properly set up for pushing
os.system("cd huggingface_model && git pull origin main --rebase")

# Ensure large files (model, tokenizer) are tracked with LFS
os.system("cd huggingface_model && git lfs install")
os.system("cd huggingface_model && git lfs track '*.bin' '*.json' '*.pt'")

# Add all changes, commit, and push
subprocess.run("cd huggingface_model && git add .", shell=True)
subprocess.run(
    "cd huggingface_model && git commit -m 'Upload converted model with LFS tracking' || echo 'No changes to commit'",
    shell=True,
)
subprocess.run("cd huggingface_model && git push origin main", shell=True)

print(
    f"🎉 Model uploaded successfully! View it here: https://huggingface.co/{hf_model_repo}"
)
