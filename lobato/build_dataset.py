import torch

from lobato import tokenizer
from lobato.model_settings import dataset

print(f"Reading dataset {dataset}...")
with open(dataset) as file:
    text = file.read()

print(f"Encoding {len(text)} characters into tokens...")
tokens = tokenizer.encode(text)

print("Building tensor...")
data = torch.tensor(tokens, dtype=torch.long)

print("Saving tensor...")
torch.save(data, "data/dataset.pt")

print("Done!")
