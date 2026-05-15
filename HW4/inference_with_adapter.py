import torch, json, os
from peft import PeftModel
from transformers import AutoProcessor
from datasets import load_dataset
from Load import model as base_model, processor

os.makedirs("experiments", exist_ok=True)

adapter_path = "./llava-finetuned/adapter"
print(f"Loading LoRA adapter from {adapter_path}...")

model = PeftModel.from_pretrained(base_model, adapter_path)
# Do NOT merge_and_unload on 4-bit models - it silently discards LoRA weights
model.eval()
print("Model with adapter loaded.\n")

# Same samples as baseline_inference.py for fair comparison
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train")
indices = [0, 10, 25, 50, 75]
samples = [dataset[i] for i in indices]

# Load baseline results for comparison
with open("experiments/baseline_results.json") as f:
    baseline_results = json.load(f)

results = []
for i, (sample, baseline) in enumerate(zip(samples, baseline_results)):
    image = sample["image"]
    question = sample["query"]
    gt_answer = sample["label"]

    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
        )

    full_output = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    answer = full_output.split("ASSISTANT:")[-1].strip()

    result = {
        "index": indices[i],
        "question": question,
        "ground_truth": gt_answer,
        "baseline_answer": baseline["baseline_answer"],
        "finetuned_answer": answer,
    }
    results.append(result)

    print(f"[Sample {i + 1}]")
    print(f"  Question     : {question}")
    print(f"  Ground Truth : {gt_answer}")
    print(f"  Baseline     : {baseline['baseline_answer']}")
    print(f"  Fine-tuned   : {answer}")
    print()

with open("experiments/finetuned_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Results saved to experiments/finetuned_results.json")
