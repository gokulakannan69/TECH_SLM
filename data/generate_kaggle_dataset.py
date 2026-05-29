import os
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Directory containing the existing RAG documents (PDFs, txt, etc.)
RAG_DOCS_DIR = Path(__file__).parents[2] / "RAG" / "documents"
# Path to the sample conversational dataset used by the SLM project
SAMPLE_DATASET = Path(__file__).parent / "sample_dataset.jsonl"
# Output file – ready to be uploaded to Kaggle as a JSONL dataset
OUTPUT_FILE = Path(__file__).parent / "slm_rag_dataset.jsonl"
# Number of synthetic documents to generate (total across all roles)
TOTAL_DOCS = 100
# Number of distinct roles
NUM_ROLES = 10
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_rag_documents() -> list[str]:
    """Collect raw text content from the RAG `documents` folder.
    Supports .txt, .md, .json, .jsonl files. Other formats are ignored.
    """
    texts = []
    for root, _, files in os.walk(RAG_DOCS_DIR):
        for fname in files:
            if fname.lower().endswith((".txt", ".md", ".json", ".jsonl")):
                fpath = Path(root) / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"[WARN] Could not read {fpath}: {e}")
    return texts

def load_sample_conversations() -> list[dict]:
    """Load the existing sample conversational dataset (JSONL)."""
    entries = []
    if not SAMPLE_DATASET.exists():
        return entries
    with open(SAMPLE_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries

def generate_role_labels(num_roles: int) -> list[str]:
    """Create placeholder role names – Role0 … Role{n-1}."""
    return [f"Role{i}" for i in range(num_roles)]

def build_documents(rag_texts: list[str], conv_entries: list[dict], roles: list[str]) -> list[dict]:
    """Combine raw RAG texts and conversational entries into a unified list of
    role‑conditioned documents.
    Each output document follows the schema:
    {
        "role": <role>,
        "content": <string>
    }
    """
    docs = []
    # Mix raw RAG snippets with conversational examples
    raw_pool = rag_texts + [json.dumps(entry, ensure_ascii=False) for entry in conv_entries]
    for _ in range(TOTAL_DOCS):
        role = random.choice(roles)
        content = random.choice(raw_pool)
        docs.append({"role": role, "content": content})
    return docs

def main():
    random.seed(42)
    rag_texts = load_rag_documents()
    conv_entries = load_sample_conversations()
    if not rag_texts and not conv_entries:
        print("[ERROR] No source data found – both RAG documents and sample dataset are missing.")
        return
    roles = generate_role_labels(NUM_ROLES)
    documents = build_documents(rag_texts, conv_entries, roles)
    # Write as JSONL – one document per line
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for doc in documents:
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[OK] Generated {len(documents)} role-conditioned documents -> {OUTPUT_FILE}")
    print("Roles used:", ", ".join(roles))

if __name__ == "__main__":
    main()
