import torch
import torch.nn as nn
from torch.nn import functional as F

from datetime import datetime

from lobato import tokenizer
from lobato.model_settings import dataset, block_size, vocab_size, n_embed, num_heads
from lobato.common import device
from lobato.model import LanguageModel

data = torch.load("data/dataset.pt")

print(f"tokens length: {len(data)}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4
training_steps = 10000
learning_rate = 1e-3
eval_interval = 100
eval_steps = 50
head_size = 16


print(f"using device: {device}")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = LanguageModel(vocab_size, n_embed, block_size, num_heads, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(training_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {step}/{training_steps}: train loss = {losses['train']:.4f}, eval loss = {losses['eval']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model_path = f"data/checkpoint-{datetime.now().timestamp()}.pt"
torch.save(model.state_dict(), model_path)
print(f"Saved checkpoint as {model_path}")
