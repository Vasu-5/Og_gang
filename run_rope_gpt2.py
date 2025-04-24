import torch
import random
import re
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config
from modeling_gpt2 import GPT2LMHeadModel  # Your custom GPT2 model with RoPE

def clean_and_trim(text, max_words=30):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    words = text.split()
    return " ".join(words[:max_words])

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config()
model = GPT2LMHeadModel(config)
model.eval()

# Load SST-2 dataset
dataset = load_dataset("csv", data_files="D:/NLP/rope_testing/sst2_dataset.csv", split="train")
label_map = {"negative": 0, "positive": 1}

# Sample 3 few-shot + 1 test example
samples = random.sample(list(dataset), 100)
few_shot_examples = samples[:99]
test_example = samples[99]

# Build prompt
few_shot_text = ""
for example in few_shot_examples:
    text = clean_and_trim(example["review"])  # Use 'review' column for the text
    sentiment = example["sentiment"]  # Use 'sentiment' column for sentiment
    few_shot_text += f"Review: {text}\nSentiment: {sentiment}\n\n"

# Add the test example without sentiment
few_shot_text += f"Review: {clean_and_trim(test_example['review'])}\nSentiment:"

# Tokenize
inputs = tokenizer(few_shot_text, return_tensors="pt", truncation=True, max_length=1024)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)

# Generate prediction
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode predicted sentiment
generated_tokens = outputs[0][input_ids.shape[-1]:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
predicted_sentiment = "positive" if "positive" in generated_text else "negative" if "negative" in generated_text else "uncertain"

# Show results
print("==== FEW-SHOT PROMPT ====")
print(few_shot_text)
print("\n==== PREDICTED SENTIMENT ====")
print(predicted_sentiment)
# Decode predicted sentiment
# generated_tokens = outputs[0][input_ids.shape[-1]:]
# generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
#
# # If the prediction is uncertain, fall back to the actual sentiment from the test example
# predicted_sentiment = "positive" if "positive" in generated_text else "negative" if "negative" in generated_text else "uncertain"
#
# # If the model is uncertain, use the original sentiment
# if predicted_sentiment == "uncertain":
#     predicted_sentiment = test_example["sentiment"]
#
# # Show results
# print("==== FEW-SHOT PROMPT ====")
# print(few_shot_text)
# print("\n==== PREDICTED SENTIMENT ====")
# print(predicted_sentiment)
