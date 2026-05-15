import os
import logging
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Create a timestamped directory for logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("./llava-finetuned/logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

os.environ["TENSORBOARD_LOGGING_DIR"] = log_dir

from datasets import load_dataset
from Load import processor
from Config import model
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

# Configure standard python logging
logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logger.info(f"Step {state.global_step}: {logs}")

# Load a tiny demo dataset (e.g., from HuggingFace)
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train[:1000]")

def format_dataset(sample):
    """
    Format dataset for SFTTrainer's prompt-completion vision-language modeling.
    SFTTrainer will automatically handle tokenization, chat templates, and
    setting prompt labels to -100 (completion-only loss).
    """
    label = (
        sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
    )
    
    # Append EOS token to the completion to teach the model when to stop generating
    label += processor.tokenizer.eos_token
    
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["query"]},
            ],
        }
    ]
    completion = [
        {"role": "assistant", "content": [{"type": "text", "text": label}]}
    ]
    
    # trl's Vision-Language SFTTrainer requires images to be in a list
    return {
        "prompt": prompt,
        "completion": completion,
        "images": [sample["image"]]
    }

# Apply formatting and remove original columns to prevent collator errors
dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

# --- Training Setup ---
# Use SFTConfig instead of TrainingArguments when using SFTTrainer in newer trl versions
training_args = SFTConfig(
    output_dir="./llava-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_dir=log_dir,
    logging_steps=10,
    report_to=["tensorboard"],
    num_train_epochs=3,
    save_strategy="no",
    max_length=1024,
    # skip_prepare_dataset=False allows SFTTrainer to natively parse prompt/completion
    dataset_kwargs={"skip_prepare_dataset": False},
)

# --- Trainer Setup ---
# By using SFTTrainer with the prompt/completion format, we can completely eliminate
# the need for a manual MultimodalCollator. The SFTTrainer will natively handle it!
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
    callbacks=[LoggingCallback()],
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save the LoRA adapter
model.save_pretrained("./llava-finetuned/adapter")
print("Adapter saved to ./llava-finetuned/adapter")
