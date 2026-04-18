import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

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
    model_kwargs={"torch_dtype": torch.float16}, # Use fp16 for LLM
    device_map="auto" 
)

# 1. Pick a Human Essay to rewrite
human_text_sample = X_val.iloc[0] # Take one sample
print(f"Original Human Text (Snippet): {human_text_sample[:200]}...")

# 2. Generate Attack (Rewrite)
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Rewrite the following essay to make it sound slightly more academic, but keep the reasoning simple:\n\n{human_text_sample}"}
]

output = generator(
    prompt, 
    max_new_tokens=512, 
    do_sample=True, 
    temperature=0.7
)

generated_text = output[0]['generated_text'][-1]['content']
print(f"\nGenerated Attack Text (Snippet): {generated_text[:200]}...")

# 3. Test against your Detector
print("\nTesting against BERT Detector...")
detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to("cuda")
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)

inputs = detector_tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")

with torch.no_grad():
    logits = detector(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)

# Label 0 = Human, Label 1 = AI
ai_prob = probabilities[0][1].item()
print(f"Detector Prediction: {'AI-Generated' if ai_prob > 0.5 else 'Human-Written'}")
print(f"Confidence (AI Class): {ai_prob:.4f}")

if ai_prob < 0.5:
    print("SUCCESS! The LLM fooled the detector.")
else:
    print("FAILED. The detector caught the LLM.")