from datasets import load_dataset
import pandas as pd

def load_ledgar_dataset():
    """
    Loads a small subset of the LEDGAR dataset for multi-label classification.
    """
    print("📥 Loading LEDGAR dataset from Hugging Face...")
    dataset = load_dataset("lex_glue", "ledgar")

    # Convert to pandas DataFrame for easier preprocessing
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    print("✅ Dataset loaded successfully!")
    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))
    print("Columns:", train_df.columns.tolist())

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_ledgar_dataset()
    print(train_df.head())
