import os, argparse
from huggingface_hub import HfApi, login
from peft import PeftConfig

def deploy(repo_id, adapter_dir, token):
    print("Logging into Hugging Face Hub...")
    login(token=token)
    api = HfApi()

    print(f"Creating/uploading model adapters to repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model")
    print(f"🎉 Adapter weights uploaded successfully! View at https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Replace these strings with your actual Hugging Face details to run directly!
    HF_REPO = "your-username/tech-tutor-slm"
    ADAPTER_DIR = "checkpoints/tech_tutor_slm/final_adapter"
    HF_TOKEN = "your_write_token_here"

    if HF_REPO != "your-username/tech-tutor-slm" and HF_TOKEN != "your_write_token_here":
        deploy(HF_REPO, ADAPTER_DIR, HF_TOKEN)
    else:
        print("Please configure HF_REPO and HF_TOKEN variables in deploy_hf.py before running.")
