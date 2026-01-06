

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL_PATH = "results/checkpoint-9375"
THRESHOLD = 0.05   # ← raise this

print(" Loading trained model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print(" Loading LEDGAR label names...")
dataset = load_dataset("lex_glue", "ledgar")
label_names = dataset["train"].features["label"].names

print(" System ready for predictions!")

while True:
    text = input("\n Enter a legal document (type 'exit' to quit):\n> ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]

    # Pair labels with probabilities
    all_preds = list(zip(label_names, probs.tolist()))
    all_preds.sort(key=lambda x: x[1], reverse=True)

    print("\n Top Model Predictions:")
    for label, score in all_preds[:10]:
        print(f"{label:<30}  {score:.4f}")

    print("\n Final Selected Labels (above threshold):")
    selected = [(l, s) for l, s in all_preds if s >= THRESHOLD]

    if not selected:
        print(" No label exceeded confidence threshold.")
    else:
        for label, score in selected:
            print(f"{label:<30}  {score:.4f}")
