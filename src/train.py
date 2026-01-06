import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
print("🔄 Loading LEDGAR dataset...")
dataset = load_dataset("lex_glue", "ledgar")

NUM_LABELS = 100

# ----------------------------
# 2. Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize_function, batched=True)

# ----------------------------
# 3. Encode labels (multi-hot)
# ----------------------------
def encode_labels(example):
    multi_hot = torch.zeros(NUM_LABELS)
    if isinstance(example["label"], list):
        for l in example["label"]:
            multi_hot[l] = 1
    else:
        multi_hot[example["label"]] = 1
    example["labels"] = multi_hot
    return example

dataset = dataset.map(encode_labels)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------
# 4. Compute class weights
# ----------------------------
print("⚖️ Computing class weights...")
label_sum = torch.zeros(NUM_LABELS)
for sample in dataset["train"]:
    label_sum += sample["labels"]

class_weights = 1.0 / (label_sum + 1e-6)
class_weights = class_weights / class_weights.mean()

# ----------------------------
# 5. Model
# ----------------------------
print("🧠 Loading Legal-BERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# ----------------------------
# 6. Custom Trainer (COMPATIBLE FIX)
# ----------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

# ----------------------------
# 7. Metrics
# ----------------------------
def compute_metrics(pred):
    probs = torch.sigmoid(torch.tensor(pred.predictions))
    preds = (probs > 0.3).int()
    labels = torch.tensor(pred.label_ids)

    f1 = f1_score(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)

    try:
        roc = roc_auc_score(labels, preds, average="micro")
    except:
        roc = 0.0

    return {"f1": f1, "accuracy": acc, "roc_auc": roc}

# ----------------------------
# 8. Training Configuration
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# ----------------------------
# 9. Trainer
# ----------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(30000)),
    eval_dataset=dataset["test"].select(range(3000)),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# ----------------------------
# 10. Train
# ----------------------------
print("🚀 Starting improved training...")
trainer.train(resume_from_checkpoint=True)

print("✅ Training finished successfully!")
