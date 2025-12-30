from transformers import AutoModelForSequenceClassification

def build_model(num_labels=100):
    print("🏗️ Initializing Legal-BERT for multi-label classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=num_labels
    )
    print("Model initialized successfully!")
    return model

if __name__ == "__main__":
    model = build_model()
