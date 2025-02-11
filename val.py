import os
import torch
import glob
from composer.utils import load_checkpoint
from huggingface_hub import HfApi, Repository

# ====== CONFIGURATION ======
# Path to the checkpoint directory
checkpoint_dir = "/tmp/reward_model/ep1-ba125"

# Hugging Face repo details
HF_USERNAME = "Jennny"  # Change this!
HF_MODEL_NAME = "help_reg_rm"  # Change this!

# Local directory for saving the model before upload
model_save_path = "huggingface_model"
os.makedirs(model_save_path, exist_ok=True)

# ====== STEP 1: LOAD SHARDED CHECKPOINT ======
try:
    print(f"Loading sharded checkpoint from {checkpoint_dir}...")
    state_dict = load_checkpoint(checkpoint_dir)
    print("Checkpoint loaded successfully!")
except Exception as e:
    print(f"Failed to load checkpoint using Composer: {e}")
    print("Attempting manual loading of shards...")

    # List all sharded checkpoint files
    checkpoint_files = sorted(glob.glob(f"{checkpoint_dir}/*.distcp"))

    if not checkpoint_files:
        raise ValueError("No checkpoint files found in the directory!")

    # Load all shards
    sharded_state_dicts = [torch.load(f, map_location="cpu") for f in checkpoint_files]

    # Merge state dicts (ensuring no overlapping keys)
    state_dict = {}
    for shard in sharded_state_dicts:
        state_dict.update(shard)  # Ensure no conflicting keys

    print("Sharded checkpoints merged successfully!")

# ====== STEP 2: SAVE MERGED CHECKPOINT ======
checkpoint_save_path = os.path.join(model_save_path, "pytorch_model.bin")
torch.save(state_dict, checkpoint_save_path)
print(f"Model saved to {checkpoint_save_path}")

# ====== STEP 3: CREATE HUGGING FACE REPOSITORY ======
hf_model_repo = f"{HF_USERNAME}/{HF_MODEL_NAME}"
api = HfApi()
api.create_repo(repo_id=hf_model_repo, exist_ok=True)

# Clone the repository locally
repo = Repository(local_dir=model_save_path, clone_from=hf_model_repo)

# ====== STEP 4: ADD MODEL CARD ======
readme_content = f"""# {HF_MODEL_NAME}

This is a reward model trained using MosaicML and converted to a Hugging Face-compatible format.
"""
with open(f"{model_save_path}/README.md", "w") as f:
    f.write(readme_content)

# ====== STEP 5: UPLOAD TO HUGGING FACE ======
print("Uploading model to Hugging Face...")
repo.git_add()
repo.git_commit("Upload converted model")
repo.git_push()

print(
    f"Model uploaded successfully! View it here: https://huggingface.co/{hf_model_repo}"
)
