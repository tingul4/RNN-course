import torch
from datasets import load_dataset
import json, os
from Load import model, processor

os.makedirs("experiments", exist_ok=True)

# Load ChartQA test split and pick 5 diverse samples
dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
# Fixed indices for reproducibility
indices = [0, 10, 25, 50, 75]
samples = [dataset[i] for i in indices]

results = []
for i, sample in enumerate(samples):
    image = sample["image"]
    question = sample["query"]
    gt_answer = sample["label"]

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=100)

    full_output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Extract only the ASSISTANT part
    answer = full_output.split("ASSISTANT:")[-1].strip()

    result = {
        "index": indices[i],
        "question": question,
        "ground_truth": gt_answer,
        "baseline_answer": answer,
    }
    results.append(result)
    print(f"[Sample {i+1}]")
    print(f"  Question   : {question}")
    print(f"  Ground Truth: {gt_answer}")
    print(f"  Baseline   : {answer}")
    print()

    # Save image for report
    img_path = f"experiments/sample_{i+1}.png"
    image.save(img_path)

# Dump results to JSON for later use in report
with open("experiments/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Results saved to experiments/baseline_results.json")