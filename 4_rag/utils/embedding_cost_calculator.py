import os
import tiktoken

# -----------------------------
# File path
# -----------------------------
file_path = os.path.join(os.path.dirname(__file__), "..", "books", "odyssey.txt")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# -----------------------------
# Read text
# -----------------------------
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# Tokenize
# -----------------------------
# cl100k_base is used for GPT-4 and GPT-3.5-turbo
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(text)
total_tokens = len(tokens)

# -----------------------------
# Estimate cost
# -----------------------------
cost_per_million_tokens = 0.02  # $0.02 per 1M tokens (example for embedding model)
estimated_cost = (total_tokens / 1_000_000) * cost_per_million_tokens

# -----------------------------
# Results
# -----------------------------
print(f"Total tokens: {total_tokens}")
print(f"Estimated cost to process: ${estimated_cost:.6f}")