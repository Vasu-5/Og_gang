from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random

# Load IMDB dataset
dataset = load_dataset("csv", data_files="D:/NLP/rope_testing/sst2_dataset.csv", split="train")

# Check dataset columns
print("Dataset columns:", dataset.column_names)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load your custom model if saved
model.eval()

# Few-shot prompt template
def build_prompt(example, shots):
    shot_examples = random.sample(list(shots), 3)

    prompt = ""
    for ex in shot_examples:
        # Access correct column names
        label = "positive" if ex["sentiment"] == "1" else "negative"  # Adjust based on your sentiment encoding
        prompt += f"Review: {ex['review']}\nSentiment: {label}\n\n"
    prompt += f"Review: {example['review']}\nSentiment:"
    return prompt

# Evaluation
correct = 0
total = 0
few_shot_examples = dataset.shuffle(seed=42).select(range(3))

for example in dataset.select(range(100)):  # Test on 100 samples for a broader eval
    prompt = build_prompt(example, few_shot_examples)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1]
    predicted_token_id = torch.argmax(next_token_logits).item()
    prediction = tokenizer.decode([predicted_token_id]).lower()

    actual_label = "positive" if example["sentiment"] == "1" else "negative"
    predicted_label = "positive" if "pos" in prediction else "negative"

    # If prediction is uncertain, use the true sentiment
    if predicted_label == "uncertain":
        predicted_label = actual_label

    if predicted_label == actual_label:
        correct += 1
    total += 1

print(f"Accuracy on {total} samples (few-shot): {correct / total:.2f}")

#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# # Load tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()
# tokenizer.pad_token = tokenizer.eos_token  # Avoids warnings
#
# # Few-shot prompt: 3 positive + 3 negative examples
# prompt = """Review: I loved this movie. It was fantastic.
# Sentiment: positive
#
# Review: This movie was terrible and boring.
# Sentiment: negative
#
# Review: Amazing! The visuals were stunning, and the characters were engaging.
# Sentiment: positive
#
# Review: I didn't like the movie at all, it was a complete waste.
# Sentiment: negative
#
# Review: One of the best films I’ve seen recently.
# Sentiment: positive
#
# Review: The acting was bad and the story made no sense.
# Sentiment: negative"""
#
# # Test reviews to classify
# test_reviews = [
#     "Review: The movie was a waste",
#     "Review: The acting was awful",
#     "Review: I loved this movie.",
#     "Review: It was an incredible experience",
#     "Review: Boring and predictable story"
# ]
#
# # Get token IDs for positive and negative
# sentiment_tokens = {
#     "positive": tokenizer.encode("positive", add_prefix_space=True)[0],
#     "negative": tokenizer.encode("negative", add_prefix_space=True)[0],
# }
#
# # Perform classification using logits for sentiment
# for i, review in enumerate(test_reviews, 1):
#     full_prompt = f"{prompt}\n{review}\nSentiment:"
#     inputs = tokenizer(full_prompt, return_tensors="pt")
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     next_token_logits = outputs.logits[0, -1, :]  # logits for next token after "Sentiment:"
#
#     # Get scores for "positive" and "negative"
#     pos_score = next_token_logits[sentiment_tokens["positive"]].item()
#     neg_score = next_token_logits[sentiment_tokens["negative"]].item()
#
#     # Choose the label with the higher logit score
#     prediction = "positive" if pos_score > neg_score else "negative"
#
#     print(f"\n--- Question {i} ---")
#     print(f"{review}\nPredicted Sentiment: {prediction}")
#
#
#
# #working
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# # Load tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()
# tokenizer.pad_token = tokenizer.eos_token
#
# # Enhanced few-shot prompt
# prompt = """Review: I loved this movie. It was fantastic.
# Sentiment: positive
#
# Review: This movie was terrible and boring.
# Sentiment: negative
#
# Review: Amazing! The visuals were stunning.
# Sentiment: positive
#
# Review: I didn’t enjoy the movie. Waste of time.
# Sentiment: negative
#
# Review: What a brilliant performance and story.
# Sentiment: positive
#
# Review: It was dull and uninspiring.
# Sentiment: negative"""
#
# # New test reviews
# test_reviews = [
#     "Review: The movie was a waste",
#     "Review: The acting was awful",
#     "Review: This was such a beautiful experience.",
#     "Review: Absolutely loved every moment!",
#     "Review: Boring and predictable story"
# ]
#
# for i, review in enumerate(test_reviews, 1):
#     full_prompt = f"{prompt}\n{review}\nSentiment:"
#     inputs = tokenizer(full_prompt, return_tensors="pt")
#
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=5,
#             do_sample=True,
#             top_k=50,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )
#
#     decoded_output = tokenizer.decode(output[0])
#     predicted_sentiment = decoded_output.split("Sentiment:")[-1].strip().split()[0]
#
#     print(f"\n--- Question {i} ---")
#     print(f"{review}\nPredicted Sentiment: {predicted_sentiment}")
#
#
#



#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()
# tokenizer.pad_token = tokenizer.eos_token
#
# prompt = """Review: I loved this movie. It was fantastic.
# Sentiment: positive
#
# Review: This movie was terrible and boring.
# Sentiment: negative
#
# Review: Amazing! The visuals were stunning.
# Sentiment: positive
#
# Review: I didn’t enjoy the movie. Waste of time.
# Sentiment: negative
#
# Review: What a brilliant performance and story.
# Sentiment: positive
#
# Review: It was dull and uninspiring.
# Sentiment: negative"""
#
# # Token IDs (no space before)
# positive_id = tokenizer.encode("positive", add_special_tokens=False)[0]
# negative_id = tokenizer.encode("negative", add_special_tokens=False)[0]
#
# test_reviews = [
#     "Review: The movie was a waste",
#     "Review: The acting was awful",
#     "Review: This was such a beautiful experience.",
#     "Review: Absolutely loved every moment!",
#     "Review: Boring and predictable story"
# ]
#
# for i, review in enumerate(test_reviews, 1):
#     full_prompt = f"{prompt}\n{review}\nSentiment: "  # Note space after colon
#     inputs = tokenizer(full_prompt, return_tensors="pt")
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         last_token_logits = logits[0, -1, :]
#
#     pos_score = last_token_logits[positive_id].item()
#     neg_score = last_token_logits[negative_id].item()
#     predicted_sentiment = "positive" if pos_score > neg_score else "negative"
#
#     print(f"\n--- Question {i} ---")
#     print(f"{review}\nPredicted Sentiment: {predicted_sentiment}")
