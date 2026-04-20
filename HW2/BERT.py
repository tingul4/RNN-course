import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一張卡，避免被其他滿載的卡或顯示卡干擾

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import softmax

# ================= CONFIGURATION =================
# Students should run this twice: once with 'bert-base-cased', once with 'bert-large-cased'
# MODEL_NAME = "bert-base-cased"
MODEL_NAME = "bert-large-cased" 

MAX_LEN = 512
BATCH_SIZE = 16 # RTX 4090 can handle 16-32 for Large, 32-64 for Base
EPOCHS = 2
# =================================================

# Load Data (Assuming a CSV with 'text' and 'label' columns)
# Students should download the dataset from Kaggle
df = pd.read_csv("dataset/train_v2_drcat_02.csv")  # Example filename
df = df[['text', 'label']] # Ensure columns exist

# Split Data
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 1. Prepare Hugging Face Dataset
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})

hf_train = Dataset.from_pandas(train_df)
hf_val = Dataset.from_pandas(val_df)

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized_train = hf_train.map(tokenize_function, batched=True)
tokenized_val = hf_val.map(tokenize_function, batched=True)

# 3. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 4. Training Arguments (Optimized for RTX 4090)
training_args = TrainingArguments(
    output_dir=f"./results_{MODEL_NAME}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,  # <--- CRITICAL for RTX 4090: Mixed Precision (Speed up + Less VRAM)
    logging_steps=50,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print(f"Starting training for {MODEL_NAME}...")
trainer.train()

pred_out = trainer.predict(tokenized_val)
logits = pred_out.predictions
labels = pred_out.label_ids
probs = softmax(logits, axis=-1)[:, 1]
auc = roc_auc_score(labels, probs)

# 6. Evaluation
results = trainer.evaluate()
print(f"Evaluation Results for {MODEL_NAME}: {results}")
print(f"ROC-AUC for {MODEL_NAME}: {auc}")

# Save the model for Part 3
model.save_pretrained(f"./saved_model_{MODEL_NAME}")
tokenizer.save_pretrained(f"./saved_model_{MODEL_NAME}")