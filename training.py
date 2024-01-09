# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: noskGPT
#     language: python
#     name: python3
# ---

# # noskGPT Training Notebook
# ---

# +
import torch

from noskgpt.model import LanguageModel
from noskgpt.tokeniser import CharToIntTokeniser
from noskgpt.training import train_model
# -

# ### Load Raw Dataset
# ---

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

print(f"Number of characters in the dataset: {len(corpus)}")

print(corpus[:400])

vocab = "".join(sorted(list(set(corpus))))
vocab_size = len(vocab)
print(f"vocab: {vocab}")
print(f"vocab size: {vocab_size}")

# ### Tokenise Dataset
# ---

# +
tokeniser = CharToIntTokeniser(corpus)

print(tokeniser.encode("My name is Nosk."))
print(tokeniser.decode(tokeniser.encode("My name is Nosk.")))
# -

data = tokeniser.encode(corpus)

# ### Partition Dataset
# ---

train_proportion = 0.9

n_train = int(train_proportion * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

# ### Create Model
# ---

block_size = 256
num_embeddings = 128
num_heads = 4
num_layers = 5
dropout = 0.1

model = LanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    num_embeddings=num_embeddings,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
)

prompt = "hello!"
model.generate(prompt=prompt, tokeniser=tokeniser, max_new_tokens=400)

# ### Train Model
# ---

batch_size = 64
learning_rate = 1e-3
max_iters = 10000
eval_iters = 100

train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    learning_rate=learning_rate,
    block_size=block_size,
    batch_size=batch_size,
    max_iters=max_iters,
    eval_iters=eval_iters
)

prompt = "\n"
model.generate(prompt=prompt, tokeniser=tokeniser, max_new_tokens=1000)

model2 = LanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    num_embeddings=num_embeddings,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
)
model2.load_state_dict(torch.load("weights_better.pth"))
model2.eval()

torch.save(model.state_dict(), "weights.pth")


