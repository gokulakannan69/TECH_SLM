import os
import torch

try:
    import gradio as gr
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError as e:
    print("\n" + "="*70)
    print("[!] [Dependency Error] Missing required Gradio or Machine Learning Libraries!")
    print(f"Details: {e}")
    print("="*70)
    print("To install all dependencies locally on your machine, run:")
    print("  pip install -r requirements.txt")
    print("\nOr open the project notebook in Google Colab to run it on a free T4 GPU!")
    print("="*70 + "\n")
    import sys
    sys.exit(1)

# Global model & tokenizer placeholders
model = None
tokenizer = None
current_model_info = "No model loaded yet."

def load_model_pipeline(model_id: str, adapter_dir: str = None):
    """Safely loads model and tokenizer on CPU or quantized GPU."""
    global model, tokenizer, current_model_info
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Gradio App] Loading {model_id} on {device.upper()}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # 4-bit Quantization Config for GPU loading
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        else:
            model_kwargs = {"device_map": "cpu"}

        # Load Base Causal LM
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            **model_kwargs
        )

        # Merge PEFT adapters if requested
        if adapter_dir and os.path.exists(adapter_dir):
            print(f"[Gradio App] Blending PEFT adapters from: {adapter_dir}")
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            current_model_info = f"Loaded Fine-Tuned Model:\nBase: {model_id}\nAdapters: {adapter_dir}"
        else:
            model = base_model
            adapter_info = f" (Adapter not found at '{adapter_dir}'!)" if adapter_dir else ""
            current_model_info = f"Loaded Base Model:\n{model_id}{adapter_info}"
            
        model.eval()
        return f"Success! {current_model_info}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Custom system prompt defining the TechTutor persona
SYSTEM_PROMPT = (
    "You are TechTutor, a friendly and pedagogical software engineering tutor. "
    "Explain concepts using: 1) An intuitive real-world analogy, "
    "2) A clean Python code example, and 3) A clear time/space complexity analysis."
)

def respond(message, chat_history, temperature, top_p, max_tokens, repetition_penalty):
    """Handles conversational turns and streams responses."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        clean_history = [m for m in chat_history if isinstance(m, dict)]
        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": "⚠️ Error: No model has been loaded yet. Please configure and load a model in the sidebar!"})
        return "", clean_history

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reconstruct conversation history including system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add previous turns and build a sanitized history list for Gradio 5
    clean_history = []
    for msg in chat_history:
        if isinstance(msg, dict):
            messages.append({"role": msg["role"], "content": msg["content"]})
            clean_history.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Convert any old tuples from the user's browser cache into dicts
            if msg[0]: 
                messages.append({"role": "user", "content": msg[0]})
                clean_history.append({"role": "user", "content": msg[0]})
            if msg[1]: 
                messages.append({"role": "assistant", "content": msg[1]})
                clean_history.append({"role": "assistant", "content": msg[1]})
            
    # Add current user prompt
    messages.append({"role": "user", "content": message})
    
    try:
        # Format conversation with prompt template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=(temperature > 0.0),
                pad_token_id=tokenizer.pad_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        response_tokens = output_ids[0][input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Append user and assistant messages as dicts
        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": response})
        return "", clean_history
    except Exception as e:
        # Record error response in dict format
        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": f"⚠️ Generation error: {str(e)}"})
        return "", clean_history

# Setup the CSS layout and design system
custom_css = """
body, .gradio-container { background-color: #ffffff !important; color: #111111 !important; }
#header {text-align: center; margin-bottom: 20px; background: white; padding: 20px;}
#header h1 {color: #dc2626; font-size: 2.5em; font-weight: 800; margin-bottom: 5px;}
#header p {font-size: 1.1em; color: #555555;}
.badge {
    background: #dc2626;
    color: white;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.85em;
    font-weight: 600;
    display: inline-block;
    margin-right: 5px;
    border: 1px solid #dc2626;
}
"""

# Build premium Gradio Block interface
# Theme and CSS config (passed to launch() for Gradio 6.0+ compatibility)
app_theme = gr.themes.Default(
    primary_hue="red", 
    secondary_hue="red", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="white",
    body_text_color="black",
    block_background_fill="white",
    block_border_color="#dc2626",
    block_label_text_color="#dc2626",
    button_primary_background_fill="#dc2626",
    button_primary_background_fill_hover="#b91c1c",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#fef2f2",
    button_secondary_text_color="#dc2626",
    button_secondary_border_color="#dc2626"
)

with gr.Blocks() as demo:
    
    # Custom Banner Header
    gr.HTML(
        """
        <div id="header">
            <h1>TechTutor SLM Hub ⚡</h1>
            <p>An interactive, premium playground for testing your fine-tuned Small Language Model (SLM).</p>
            <div style="margin-top: 10px;">
                <span class="badge">Model: TinyLlama-1.1B</span>
                <span class="badge">Fine-Tuning: QLoRA</span>
                <span class="badge">Persona: Pedagogical Coding Tutor</span>
            </div>
        </div>
        """
    )
    
    with gr.Accordion("⚙️ Advanced Settings & Model Configuration", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Model Initialization")
                model_id_input = gr.Textbox(
                    label="Base Model ID", 
                    value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    placeholder="e.g. microsoft/Phi-3-mini-4k-instruct"
                )
                adapter_dir_input = gr.Textbox(
                    label="PEFT Adapter Directory (Optional)", 
                    value="checkpoints/tech_tutor_slm/final_adapter",
                    placeholder="Path to LoRA checkpoints"
                )
                
                load_btn = gr.Button("🔌 Initialize Model", variant="primary")
                status_output = gr.Textbox(label="System Status", value="Not initialized", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Hyperparameter Tuner")
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.2, value=0.7, step=0.05, 
                    label="Temperature (Creativity)"
                )
                top_p_slider = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.9, step=0.05, 
                    label="Top-P (Nucleus Sampling)"
                )
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64, 
                    label="Max New Tokens"
                )
                rep_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=1.5, value=1.15, step=0.05, 
                    label="Repetition Penalty"
                )
                
    # Main Chat Interface
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="TechTutor Interactive AI", height=500)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Ask TechTutor a programming question...",
                    placeholder="e.g. Explain how a hash map works.",
                    scale=4
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                
            with gr.Row():
                clear_btn = gr.ClearButton([msg_input, chatbot], value="🧹 Reset Chat")
                
            # Quick suggestion buttons
            gr.Markdown("#### ✨ Quick Suggestion Starters")
            with gr.Row():
                btn_rec = gr.Button("⚙️ Explain Recursion")
                btn_hash = gr.Button("🔑 Explain Hash Maps")
                btn_api = gr.Button("👶 Explain APIs to a 5-year-old")
                
    # Bind load model button
    load_btn.click(
        fn=load_model_pipeline, 
        inputs=[model_id_input, adapter_dir_input], 
        outputs=status_output
    )
    
    # Bind prompt submission
    submit_btn.click(
        fn=respond, 
        inputs=[msg_input, chatbot, temperature_slider, top_p_slider, max_tokens_slider, rep_penalty_slider], 
        outputs=[msg_input, chatbot]
    )
    msg_input.submit(
        fn=respond, 
        inputs=[msg_input, chatbot, temperature_slider, top_p_slider, max_tokens_slider, rep_penalty_slider], 
        outputs=[msg_input, chatbot]
    )
    
    # Bind quick suggestion buttons to prefill prompt and trigger response
    def set_prompt_and_run(txt):
        return txt
        
    btn_rec.click(fn=lambda: "Explain recursion with an analogy and a python factorial example.", outputs=msg_input)
    btn_hash.click(fn=lambda: "What is a Hash Map and how does it achieve O(1) average lookup times?", outputs=msg_input)
    btn_api.click(fn=lambda: "Can you explain what APIs are, and make it so a 5-year-old can understand?", outputs=msg_input)

if __name__ == "__main__":
    # Check if adapter folder exists locally to auto-load on start
    default_adapter = "checkpoints/tech_tutor_slm/final_adapter"
    adapter_path = default_adapter if os.path.exists(default_adapter) else None
    
    # Automatically load the model at startup so the UI doesn't need manual init
    load_msg = load_model_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path)
    print(load_msg)
    
    # Boot model silently on server startup to speed up launch
    device_status = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(f"System boot started. Active processing unit: {device_status}")
    
    # Launch Gradio on all interfaces (0.0.0.0) at port 7860 for local access.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, theme=app_theme, css=custom_css)
