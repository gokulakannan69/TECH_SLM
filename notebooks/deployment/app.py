import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model = None
tokenizer = None

def load_model_pipeline(model_id, adapter_dir):
    global model, tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if device == "cuda" else None
    model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"} if bnb_config else {"device_map": "cpu"}

    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, **model_kwargs)
    if adapter_dir and os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()
    return f"Successfully loaded model: {model_id} | Adapter loaded: {os.path.exists(adapter_dir) if adapter_dir else False}"

SYSTEM_PROMPT = "You are TechTutor, a friendly and pedagogical software engineering tutor. Explain concepts using: 1) An intuitive real-world analogy, 2) A clean Python code example, and 3) A clear time/space complexity analysis."

def respond(message, chat_history, temperature, top_p, max_tokens):
    global model, tokenizer
    if model is None:
        return "", chat_history + [(message, "⚠️ Error: Initialize the model in the sidebar first!")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_turn, assistant_turn in chat_history:
        if user_turn: messages.append({"role": "user", "content": user_turn})
        if assistant_turn and not assistant_turn.startswith("⚠️"): messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": message})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo")) as demo:
    gr.HTML("<div style='text-align: center;'><h1>TechTutor SLM Hub ⚡</h1><p>A high-fidelity QLoRA-based playground.</p></div>")
    with gr.Row():
        with gr.Column(scale=1):
            model_id = gr.Textbox(label="Model ID", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            adapter_dir = gr.Textbox(label="Adapter Path", value="checkpoints/tech_tutor_slm/final_adapter")
            load_btn = gr.Button("🔌 Initialize Model", variant="primary")
            status = gr.Textbox(label="Status", value="Not loaded", interactive=False)
            temp = gr.Slider(0.0, 1.2, 0.7, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, 0.9, label="Top-P")
            max_tokens = gr.Slider(64, 1024, 512, step=64, label="Max New Tokens")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Ask TechTutor a coding question...")
            with gr.Row():
                submit = gr.Button("Send 🚀", variant="primary")
                clear = gr.ClearButton([msg, chatbot], value="🧹 Clear")

    load_btn.click(load_model_pipeline, [model_id, adapter_dir], status)
    submit.click(respond, [msg, chatbot, temp, top_p, max_tokens], [msg, chatbot])
    msg.submit(respond, [msg, chatbot, temp, top_p, max_tokens], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
