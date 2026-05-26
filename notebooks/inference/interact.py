import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def compare_models(prompt: str):
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "checkpoints/tech_tutor_slm/final_adapter"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure 4-bit load if GPU exists
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if device == "cuda" else None
    model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"} if bnb_config else {"device_map": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base model
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, **model_kwargs)

    # Set system prompt
    messages = [
        {"role": "system", "content": "You are TechTutor, a friendly and pedagogical software engineering tutor. Explain concepts using: 1) An intuitive real-world analogy, 2) A clean Python code example, and 3) A clear time/space complexity analysis."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # 1. Base Model Generation
    print("\n--- RUNNING BASE MODEL GENERATION ---")
    with torch.no_grad():
        outputs_base = base_model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    response_base = tokenizer.decode(outputs_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(response_base)

    # 2. Fine-Tuned Model Generation
    print("\n--- RUNNING FINE-TUNED MODEL GENERATION (QLoRA) ---")
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    with torch.no_grad():
        outputs_ft = ft_model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    response_ft = tokenizer.decode(outputs_ft[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(response_ft)

if __name__ == "__main__":
    compare_models("What is a Stack and how does it work?")
