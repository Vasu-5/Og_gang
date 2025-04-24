from datasets import load_dataset
dataset = load_dataset("csv", data_files="D:/NLP/rope_testing/sst2_dataset.csv", split="train")

print(dataset.column_name   s)
print(dataset.column_names)

