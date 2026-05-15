import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TENSORBOARD_LOGGING_DIR"] = "./llava-finetuned/logs"

from datasets import load_dataset
from Load import processor
from Config import model
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load a tiny demo dataset (e.g., from HuggingFace)
# Students should implement a proper formatting function here
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train[:1000]")


class ChartQADataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # ChartQA label is a list of acceptable answers; take the first
        label = (
            sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
        return {"conversation": conversation, "image": sample["image"]}


@dataclass
class MultimodalCollator:
    processor: Any

    def __call__(self, features):
        # tokenize=False returns a string, not token IDs
        texts = [
            self.processor.apply_chat_template(
                f["conversation"], add_generation_prompt=False, tokenize=False
            )
            + self.processor.tokenizer.eos_token
            for f in features
        ]
        images = [f["image"] for f in features]

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask prompt tokens: only compute loss on assistant response
        # Build prompt-only text to find where assistant response starts
        for i, f in enumerate(features):
            prompt_text = self.processor.apply_chat_template(
                f["conversation"][:-1],  # exclude assistant turn
                add_generation_prompt=True,
                tokenize=False,
            )
            # 必須同時傳入影像給 processor，才能精準算出 <image> 展開為影像 embedding 後的實際 Token 數量
            prompt_inputs = self.processor(
                text=prompt_text, images=f["image"], return_tensors="pt",
                truncation=True, max_length=1024
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch


# --- Training Setup ---
training_args = TrainingArguments(
    output_dir="./llava-finetuned",  # stores training state / logs
    per_device_train_batch_size=4,  # 4090 can handle 4-8 depending on image resolution
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    report_to="tensorboard",
    num_train_epochs=3,
    save_strategy="no",
    remove_unused_columns=False,  # our collator needs 'conversation' and 'image' keys
)

# For HW, students can use the standard Trainer if they handle data collation manually,
# or use SFTTrainer from TRL which is easier for chat formats.

# --- Trainer Setup ---

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ChartQADataset(dataset),
    data_collator=MultimodalCollator(processor),
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save the LoRA adapter
model.save_pretrained("./llava-finetuned/adapter")
print("Adapter saved to ./llava-finetuned/adapter")
