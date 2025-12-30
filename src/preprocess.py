from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset

def preprocess_data():
    print("📥 Loading LEDGAR dataset...")
    dataset = load_dataset("lex_glue", "ledgar")

    # Initialize Legal-BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    print("🔄 Tokenizing dataset... (this may take a few minutes)")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    print(" Tokenization complete!")
    print(tokenized_dataset)

    return tokenized_dataset

if __name__ == "__main__":
    preprocess_data()
