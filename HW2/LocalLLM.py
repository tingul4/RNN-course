# %%
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# Clean up memory from Part 2
import gc

torch.cuda.empty_cache()
gc.collect()

# ================= CONFIGURATION =================
# Use a high-quality instruction model.
# Students might need to login via `huggingface-cli login` for Llama-3 access.
GEN_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Alternative (No token needed): "mistralai/Mistral-7B-Instruct-v0.3"

# Load your best detector from Part 2
DETECTOR_PATH = "./saved_model_bert-base-cased"
# =================================================

print("Loading LLM for generation...")
# device_map="auto" will automatically use the RTX 4090
generator = pipeline(
    "text-generation",
    model=GEN_MODEL_NAME,
    model_kwargs={"dtype": torch.float16},
    device_map="auto",
)

# Load Data (Assuming a CSV with 'text' and 'label' columns)
# Students should download the dataset from Kaggle
df = pd.read_csv("dataset/train_v2_drcat_02.csv")  # Example filename
df = df[["text", "label"]]  # Ensure columns exist

# Split Data
X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Load detector once to avoid repeated model initialization inside loop
print("Loading BERT detector...")
detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to("cuda")
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)

# Sample only human-written validation essays (label=0)
human_indices = [idx for idx, label in enumerate(y_val.tolist()) if label == 0]
sample_size = min(10, len(human_indices))

for i in random.sample(human_indices, sample_size):
    # 1. Pick 10 Human Essay to rewrite
    human_text_sample = X_val.iloc[i]
    print(f"Original Human Text (Snippet): {human_text_sample[:200]}...")

    # 2. Generate Attack (Rewrite)
    prompt = [
        {
            "role": "system",
            "content": "You are a text rewriting engine. Output ONLY the rewritten text. Do not include any conversational filler, greetings, or acknowledgments.",
        },
        {
            "role": "user",
            "content": f"Rewrite the essay so it sounds like a real high school student wrote it. Use simple wording, natural phrasing, slightly imperfect style or incorrect grammar:\n\n{human_text_sample}",
        },
    ]

    output = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.9)

    generated_text = output[0]["generated_text"][-1]["content"]
    print(f"\n[no. {i}] Generated Attack Text (Snippet): {generated_text[:200]}...")

    # 3. Test against your Detector
    print(f"\n[no. {i}] Testing against BERT Detector...")

    inputs = detector_tokenizer(
        generated_text, return_tensors="pt", truncation=True, max_length=512
    ).to("cuda")

    with torch.no_grad():
        logits = detector(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)

    # Label 0 = Human, Label 1 = AI
    ai_prob = probabilities[0][1].item()
    print(
        f"[no. {i}] Detector Prediction: {'AI-Generated' if ai_prob > 0.5 else 'Human-Written'}"
    )
    print(f"[no. {i}] Confidence (AI Class): {ai_prob:.4f}")

    if ai_prob < 0.5:
        print(f"[no. {i}] SUCCESS! The LLM fooled the detector.")
    else:
        print(f"[no. {i}] FAILED. The detector caught the LLM.")
    print("-" * 80)
