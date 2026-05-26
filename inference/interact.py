import os
import torch
import argparse

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError as e:
    print("\n" + "="*70)
    print("[!] [Dependency Error] Missing required Machine Learning Libraries!")
    print(f"Details: {e}")
    print("="*70)
    print("To install all dependencies locally on your machine, run:")
    print("  pip install -r requirements.txt")
    print("\nOr open the project notebook in Google Colab to run it on a free T4 GPU!")
    print("="*70 + "\n")
    import sys
    sys.exit(1)

def load_inference_model(model_id: str, adapter_dir: str = None):
    """Loads a base causal LM and optionally applies LoRA adapters."""
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure correct padding side for generation
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device.upper()}")

    # 4-bit config for efficient GPU loading
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("Using 4-bit QLoRA loading configuration to conserve VRAM.")
    else:
        bnb_config = None
        print("Running on CPU. No 4-bit quantization will be applied (bitsandbytes is CUDA-only).")

    # Load base model
    model_kwargs = {}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        **model_kwargs
    )

    if adapter_dir and os.path.exists(adapter_dir):
        print(f"Applying fine-tuned LoRA adapters from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print("Adapters loaded and combined successfully.")
    else:
        if adapter_dir:
            print(f"Warning: Adapter path '{adapter_dir}' not found. Defaulting to raw base model.")
        model = base_model
        print("Using raw base model (no adapters applied).")

    model.eval()
    return model, tokenizer

def run_chat_loop(model, tokenizer, args):
    """Main interactive terminal chat loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*50)
    print("      TechTutor SLM Interactive CLI Portal      ")
    print(f"      Base Model: {args.model_id}")
    print(f"      Adapter:    {args.adapter_dir if args.adapter_dir else 'None (Raw Base)'}")
    print("="*50)
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'settings' to view/edit generation parameters.")
    print("="*50 + "\n")

    # Interactive generation settings
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0.0,
        "repetition_penalty": 1.15
    }

    # Custom persona system prompt
    system_prompt = (
        "You are TechTutor, a friendly and pedagogical software engineering tutor. "
        "Explain concepts using: 1) An intuitive real-world analogy, "
        "2) A clean Python code example, and 3) A clear time/space complexity analysis."
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Session terminated. Happy coding!")
                break
                
            if user_input.lower() == "settings":
                print("\nCurrent Settings:")
                for k, v in gen_config.items():
                    print(f"  {k}: {v}")
                
                # Simple CLI adjustment
                new_temp = input("Enter new temperature (or press Enter to keep): ").strip()
                if new_temp:
                    try:
                        gen_config["temperature"] = float(new_temp)
                        gen_config["do_sample"] = float(new_temp) > 0.0
                        print(f"Temperature updated to {new_temp}.")
                    except ValueError:
                        print("Invalid number. Keeping previous settings.")
                continue

            # Format using standard chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            print("\nTechTutor: Thinking...", end="", flush=True)

            # Generate output
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_p=gen_config["top_p"],
                    do_sample=gen_config["do_sample"],
                    repetition_penalty=gen_config["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Strip prompt tokens from output
            input_length = inputs.input_ids.shape[1]
            generated_tokens = output_ids[0][input_length:]
            
            # Decode output
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Print response
            print("\r" + " "*20 + "\r" + "TechTutor:\n" + response + "\n")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interact with the fine-tuned Small Language Model")
    
    # Model configs
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model Hugging Face ID")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/tech_tutor_slm/final_adapter", help="Directory where adapter weights are saved")
    
    # Default Generation configs
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    
    args = parser.parse_args()
    
    # Check if adapter folder exists, print warning if not
    use_adapter = args.adapter_dir
    if not os.path.exists(args.adapter_dir):
        print(f"Info: Adapter directory not found at '{args.adapter_dir}'.")
        print("We will run the raw base model with the tutor system prompt to see the difference.")
        use_adapter = None
        
    model, tokenizer = load_inference_model(args.model_id, adapter_dir=use_adapter)
    run_chat_loop(model, tokenizer, args)
