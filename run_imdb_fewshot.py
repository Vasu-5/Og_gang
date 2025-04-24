# from datasets import load_dataset
# from transformers import GPT2Tokenizer
# from modeling_gpt2 import GPT2LMHeadModel
# import torch
# import random
# import numpy as np
#
# # Load IMDB dataset
# dataset = load_dataset("imdb", split="test[:100]")  # Smaller subset for quick testing
#
# # Load tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load your custom model if saved
# model.eval()
#
#
# # Few-shot prompt template
# def build_prompt(example, shots):
#     #shot_examples = random.sample(shots, 3)
#     shot_examples = random.sample(list(shots), 3)
#
#     prompt = ""
#     for ex in shot_examples:
#         label = "positive" if ex["label"] == 1 else "negative"
#         prompt += f"Review: {ex['text']}\nSentiment: {label}\n\n"
#     prompt += f"Review: {example['text']}\nSentiment:"
#     return prompt
#
#
# # Evaluation
# correct = 0
# total = 0
# few_shot_examples = dataset.shuffle(seed=42).select(range(3))
#
# for example in dataset.select(range(100)):  # Test on 20 samples
#     prompt = build_prompt(example, few_shot_examples)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     next_token_logits = outputs.logits[0, -1]
#     predicted_token_id = torch.argmax(next_token_logits).item()
#     prediction = tokenizer.decode([predicted_token_id]).lower()
#
#     actual_label = "positive" if example["label"] == 1 else "negative"
#     predicted_label = "positive" if "pos" in prediction else "negative"
#
#     if predicted_label == actual_label:
#         correct += 1
#     total += 1
#
# print(f"Accuracy on {total} IMDB samples (few-shot): {correct / total:.2f}")



import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer.pad_token = tokenizer.eos_token

prompt = """Review: I loved this movie. It was fantastic.
Sentiment: positive

Review: This movie was terrible and boring.
Sentiment: negative

Review: Amazing! The visuals were stunning.
Sentiment: positive

Review: I didn’t enjoy the movie. Waste of time.
Sentiment: negative

Review: What a brilliant performance and story.
Sentiment: positive

Review: It was dull and uninspiring.
Sentiment: negative"""

# Token IDs (no space before)
positive_id = tokenizer.encode("positive", add_special_tokens=False)[0]
negative_id = tokenizer.encode("negative", add_special_tokens=False)[0]

test_reviews = [
    "Review: The movie was a waste",
    "Review: The acting was awful",
    "Review: This was such a beautiful experience.",
    "Review: Absolutely loved every moment!",
    "Review: Boring and predictable story"
]

for i, review in enumerate(test_reviews, 1):
    full_prompt = f"{prompt}\n{review}\nSentiment: "  # Note space after colon
    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]

    pos_score = last_token_logits[positive_id].item()
    neg_score = last_token_logits[negative_id].item()
    predicted_sentiment = "positive" if pos_score > neg_score else "negative"

    print(f"\n--- Question {i} ---")
    print(f"{review}\nPredicted Sentiment: {predicted_sentiment}")
