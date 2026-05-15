import json
import os
import glob
from datasets import load_dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def is_correct(pred_str, gt_list):
    pred = pred_str.strip()
    for gt_item in gt_list:
        gt = gt_item.strip()
        # Try relaxed numeric parse
        try:
            pred_num = float(pred)
            gt_num = float(gt)
            if abs(pred_num - gt_num) / max(abs(gt_num), 1.0) <= 0.05:
                return True
        except ValueError:
            pass
        # Fallback to case-insensitive exact string match
        if pred.lower() == gt.lower():
            return True
    return False


def evaluate():
    # 1. Load finetuned results
    if not os.path.exists("experiments/finetuned_results.json"):
        print("experiments/finetuned_results.json not found!")
        return

    with open("experiments/finetuned_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    # 2. Compute accuracy & Print table
    print("Index | Question | GT | Baseline | Fine-tuned | Baseline✓ | FT✓")
    print("-" * 80)
    baseline_correct_count = 0
    ft_correct_count = 0

    for res in results:
        idx = res["index"]
        question = res["question"]
        gt = res["ground_truth"]
        baseline_ans = res["baseline_answer"]
        ft_ans = res.get("finetuned_answer", "")

        base_correct = is_correct(baseline_ans, gt)
        ft_correct = is_correct(ft_ans, gt)

        baseline_correct_count += int(base_correct)
        ft_correct_count += int(ft_correct)

        b_mark = "✓" if base_correct else "✗"
        f_mark = "✓" if ft_correct else "✗"

        # format column prints beautifully
        print(
            f"{idx:5} | {question[:20]:20} | {str(gt)[:10]:10} | {baseline_ans[:10]:10} | {ft_ans[:10]:10} | {b_mark:9} | {f_mark:3}"
        )

    print("-" * 80)
    print(f"Baseline Accuracy: {baseline_correct_count}/{len(results)}")
    print(f"Fine-tuned Accuracy: {ft_correct_count}/{len(results)}")

    # 3. Save 5 test images
    print("\nSaving test images...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    indices = [0, 10, 25, 50, 75]
    for i, idx in enumerate(indices):
        img = dataset[idx]["image"]
        img.save(f"sample_{i + 1}.png")
        print(f"Saved sample_{i + 1}.png")

    # 4. Plot loss curve
    print("\nPlotting loss curve...")
    log_dir = "llava-finetuned/logs"
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} not found!")
        return

    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print("No tfevents file found in log directory!")
        return

    event_file = event_files[0]
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract train/loss
    if "train/loss" in ea.Tags()["scalars"]:
        loss_events = ea.Scalars("train/loss")
        steps = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]

        plt.figure(figsize=(10, 6))
        plt.plot(
            steps, losses, marker="o", linestyle="-", color="b", label="Training Loss"
        )
        plt.title("Training Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig("loss_curve.png")
        print("Saved loss_curve.png")
    else:
        print("No 'train/loss' found in tfevents scalars!")
        print("Available tags:", ea.Tags()["scalars"])


if __name__ == "__main__":
    evaluate()
