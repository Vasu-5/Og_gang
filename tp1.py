import pandas as pd
from datasets import load_dataset

# Load the SST-2 dataset (train split as example)
dataset = load_dataset("stanfordnlp/sst2", split="train")

# Convert to a DataFrame and map label to string
df = pd.DataFrame(dataset)
df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})

# Save as CSV
save_path = r"D:\NLP\rope_testing\sst2_dataset.csv"
df[["sentence", "sentiment"]].to_csv(save_path, index=False)

print(f"SST-2 dataset saved to: {save_path}")
