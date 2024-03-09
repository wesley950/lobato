import pickle

from .common import get_most_common_byte_pair, merge
from model_settings import merge_count

with open("data/unb.txt", "r") as file:
    text = file.read()

tokens = text.encode()
tokens = list(map(int, tokens))


merges = {}

for step in range(merge_count):
    step_idx = step + 256
    stats = get_most_common_byte_pair(tokens)
    most_common = max(stats, key=stats.get)
    tokens = merge(tokens, most_common, step_idx)
    merges[most_common] = step_idx

with open("data/merges.pkl", "wb") as file:
    pickle.dump(merges, file)


vocab = {idx: bytes([idx]) for idx in range(256)}
for (po, pi), idx in merges.items():
    vocab[idx] = vocab[po] + vocab[pi]

with open("data/vocab.pkl", "wb") as file:
    pickle.dump(vocab, file)
