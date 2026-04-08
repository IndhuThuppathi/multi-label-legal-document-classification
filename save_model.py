from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_path = "results/checkpoint-9375"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

print("✅ Model saved correctly for classification")