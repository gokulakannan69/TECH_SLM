import os
import argparse

try:
    from huggingface_hub import HfApi, login
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print("\n" + "="*70)
    print("[!] [Dependency Error] Missing required Hugging Face Hub or Machine Learning Libraries!")
    print(f"Details: {e}")
    print("="*70)
    print("To install all dependencies locally on your machine, run:")
    print("  pip install -r requirements.txt")
    print("\nOr open the project notebook in Google Colab to run it on a free T4 GPU!")
    print("="*70 + "\n")
    import sys
    sys.exit(1)

def deploy_model_and_space(args):
    """Handles authentication and deployment of LoRA adapters and Gradio Space to Hugging Face Hub."""
    
    print("\n" + "="*60)
    print("      Hugging Face Hub SLM Deployment Portal      ")
    print("="*60)
    
    # Step 1: Authentication
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n[Error] Hugging Face Access Token is required!")
        print("Please obtain a Write-access token from https://huggingface.co/settings/tokens")
        print("And pass it via --token argument or set the HF_TOKEN environment variable.")
        return
        
    print("\n[Step 1] Authenticating with Hugging Face Hub...")
    try:
        login(token=hf_token)
        print("Successfully authenticated!")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    api = HfApi()

    # Step 2: Push fine-tuned LoRA adapter to Hub
    if args.adapter_dir and os.path.exists(args.adapter_dir):
        print(f"\n[Step 2] Pushing LoRA adapters from '{args.adapter_dir}' to Model Repository '{args.repo_id}'...")
        try:
            # Create repo if it doesn't exist
            print(f"Ensuring repository '{args.repo_id}' exists...")
            api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
            
            # Load config to verify
            config = PeftConfig.from_pretrained(args.adapter_dir)
            
            # Upload files
            api.upload_folder(
                folder_path=args.adapter_dir,
                repo_id=args.repo_id,
                repo_type="model"
            )
            print(f"🎉 Success! Fine-tuned adapters pushed. View at: https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"Failed to upload model adapters: {e}")
    else:
        print(f"\n[Step 2] Skipped adapter uploading. (Adapter path '{args.adapter_dir}' not found.)")

    # Step 3: Deploy Gradio App to Hugging Face Spaces
    if args.deploy_space:
        space_repo = args.space_id or f"{args.repo_id}-space"
        print(f"\n[Step 3] Creating Hugging Face Space for Gradio Interface: '{space_repo}'...")
        try:
            # Create Space repository
            api.create_repo(
                repo_id=space_repo,
                repo_type="space",
                space_sdk="gradio",
                exist_ok=True
            )
            
            # Write a README.md metadata block for the Space (Hugging Face requirements)
            space_readme = (
                "---\n"
                "title: TechTutor SLM Hub\n"
                "emoji: ⚡\n"
                "colorFrom: teal\n"
                "colorTo: indigo\n"
                "sdk: gradio\n"
                "sdk_version: 4.26.0\n"
                "app_file: app.py\n"
                "pinned: false\n"
                "license: apache-2.0\n"
                "---\n\n"
                "# TechTutor SLM Space\n"
                "This Space hosts the Gradio interface for interacting with our fine-tuned QLoRA TinyLlama-1.1B TechTutor SLM.\n"
            )
            
            # Create a temporary directory structure for space files
            temp_space_dir = "temp_hf_space"
            os.makedirs(temp_space_dir, exist_ok=True)
            
            # Write metadata readme
            with open(os.path.join(temp_space_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(space_readme)
                
            # Copy app.py into the space directory
            src_app = "deployment/app.py"
            dst_app = os.path.join(temp_space_dir, "app.py")
            if os.path.exists(src_app):
                with open(src_app, "r", encoding="utf-8") as f_src:
                    app_code = f_src.read()
                
                # Replace default adapter path to point to our newly uploaded HF model repo
                # This makes the Space run completely independently in the cloud!
                app_code = app_code.replace(
                    'value="checkpoints/tech_tutor_slm/final_adapter"',
                    f'value="{args.repo_id}"'
                )
                
                with open(dst_app, "w", encoding="utf-8") as f_dst:
                    f_dst.write(app_code)
            else:
                print("[Error] Source app.py not found at 'deployment/app.py'!")
                return
                
            # Write requirements.txt for the Space
            with open(os.path.join(temp_space_dir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write("torch\ntransformers\npeft\nbitsandbytes\naccelerate\n")

            # Upload Space files
            print("Uploading Space files to Hugging Face Spaces...")
            api.upload_folder(
                folder_path=temp_space_dir,
                repo_id=space_repo,
                repo_type="space"
            )
            
            # Clean up temp files
            for file in ["README.md", "app.py", "requirements.txt"]:
                filepath = os.path.join(temp_space_dir, file)
                if os.path.exists(filepath):
                    os.remove(filepath)
            os.rmdir(temp_space_dir)
            
            print(f"🎉 Success! Gradio app deployed to Hugging Face Spaces.")
            print(f"View and interact with your app at: https://huggingface.co/spaces/{space_repo}")
            
        except Exception as e:
            print(f"Failed to deploy Gradio Space: {e}")
            
    print("\n" + "="*60)
    print("      Deployment Operations Complete!      ")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SLM weights and Gradio Interface to Hugging Face Hub")
    
    # Repo configurations
    parser.add_argument("--repo_id", type=str, required=True, help="HF Model repository name (e.g., 'username/tech-tutor-slm')")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/tech_tutor_slm/final_adapter", help="Directory of saved LoRA adapter checkpoint")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write-Access Token")
    
    # Space deployment
    parser.add_argument("--deploy_space", action="store_true", help="Deploy the Gradio Chatbot to HF Spaces")
    parser.add_argument("--space_id", type=str, default=None, help="HF Space repository name (e.g., 'username/tech-tutor-space')")
    
    args = parser.parse_args()
    deploy_model_and_space(args)
