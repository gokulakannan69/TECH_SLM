# ⚡ TechTutor SLM Project

TechTutor SLM is a Small Language Model fine-tuning project built using TinyLlama, Hugging Face, QLoRA, and Gradio. The project works by training a lightweight AI model using custom datasets so the model can answer technical questions like a coding tutor.

## 🛠 Required Installations

• Python 3.9+
• PyTorch
• Transformers
• PEFT / QLoRA
• TRL
• Gradio
• Datasets
• Accelerate
• BitsAndBytes

Install all dependencies:

```bash id="h3rx0y"
pip install -r requirements.txt
```

## 🚀 How The Project Works

1. Dataset is prepared using JSONL files
2. TinyLlama base model is loaded
3. QLoRA fine-tuning is applied
4. The model trains using custom instruction datasets
5. Adapter weights are saved after training
6. The trained model is tested using CLI and Gradio UI
7. The application can be deployed using Hugging Face Spaces

## ▶️ How To Run The Project

Create virtual environment:

```bash id="1k4qzs"
python -m venv venv
```

Activate environment (Windows PowerShell):

```powershell id="7xq4mu"
.\venv\Scripts\Activate.ps1
```

Prepare dataset:

```bash id="px1a5j"
python data/prepare_dataset.py
```

Start model training:

```bash id="2xk7jw"
python training/train.py
```

Run inference:

```bash id="r8my6v"
python inference/interact.py
```

Launch web application:

```bash id="0p4zfw"
python deployment/app.py
```

Open browser:

```bash id="w9jq2e"
http://localhost:7860
```

Deploy to Hugging Face:

```bash id="z6tv1k"
python deployment/deploy_hf.py
```
