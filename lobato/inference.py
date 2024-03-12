import sys

import torch

from lobato import tokenizer
from lobato.common import device
from lobato.model import LanguageModel
from lobato.model_settings import block_size, n_embed, num_heads, vocab_size

model = LanguageModel(vocab_size, n_embed, block_size, num_heads, device)
model.load_state_dict(torch.load(sys.argv[1]))
model = model.to(device)

context = torch.tensor(tokenizer.encode(sys.argv[3]), dtype=torch.long, device=device)
context = context.view((context.size()[0], 1))
context1 = torch.zeros((1, 1), dtype=torch.long, device=device)
print(context.size())
print(context1.size())

output = model.generate(
    context,
    max_new_tokens=int(sys.argv[2]),
)[0].tolist()
output = tokenizer.decode(output)
print(output)
