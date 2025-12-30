# src/train.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
print("🚀 Loading dataset...")
dataset = load_dataset("lex_glue", "ledgar")

NUM_LABELS = 100

# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "nlpaueb/legal-bert-base-uncased"
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize_function, batched=True)

# --------------------------------------------------
# 3. Convert labels → multi-hot vectors
# --------------------------------------------------
def encode_labels(example):
    multi_hot = torch.zeros(NUM_LABELS)

    if isinstance(example["label"], list):
        for l in example["label"]:
            multi_hot[l] = 1
    else:
        multi_hot[example["label"]] = 1

    example["labels"] = multi_hot.tolist()
    return example

dataset = dataset.map(encode_labels)

# remove unused columns
dataset = dataset.remove_columns(
    [col for col in dataset["train"].column_names
     if col not in ["input_ids", "attention_mask", "labels"]]
)

dataset.set_format(type="torch")

# --------------------------------------------------
# 4. Model
# --------------------------------------------------
print("🚀 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# --------------------------------------------------
# 5. Metrics
# --------------------------------------------------
def compute_metrics(pred):
    logits = torch.tensor(pred.predictions)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    labels = torch.tensor(pred.label_ids)

    f1 = f1_score(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)

    try:
        roc = roc_auc_score(labels, preds, average="micro")
    except:
        roc = 0.0

    return {
        "f1": f1,
        "accuracy": acc,
        "roc_auc": roc
    }

# --------------------------------------------------
# 6. Training Arguments (COMPATIBLE)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",      # ⚠️ correct for your version
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=50,
    report_to="none"
)

# --------------------------------------------------
# 7. Trainer (NO CUSTOM LOSS NEEDED)
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=dataset["test"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------
print("🎯 Starting training...")
trainer.train()
print("✅ Training completed successfully!")
