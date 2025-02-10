import torch
from huggingface_hub import HfApi, HfFolder, Repository

# Define paths
checkpoint_path = "/tmp/reward_model/ep1-ba125/__0_0.distcp"
hf_model_repo = "Jennny/help_reg_rm"

# Load model state dict
state_dict = torch.load(checkpoint_path, map_location="cpu")

# If needed, define the model class and load state
# model = YourModelClass()
# model.load_state_dict(state_dict)

# Save the model in Hugging Face's format
model_save_path = "huggingface_model"
torch.save(state_dict, f"{model_save_path}/pytorch_model.bin")

# Create repository and push to Hugging Face Hub
api = HfApi()
api.create_repo(repo_id=hf_model_repo, exist_ok=True)

repo = Repository(local_dir=model_save_path, clone_from=hf_model_repo)
repo.git_add()
repo.git_commit("Upload trained model checkpoint")
repo.git_push()
