import pickle, json
import regex as re

from .common import get_stats, merge, pattern
from .model_settings import merge_count, dataset

with open(dataset, "r") as file:
    text = file.read()

chunks = list(set(re.findall(pattern, text, flags=re.IGNORECASE)))
tokens = [list(chunk.encode()) for chunk in chunks]

merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}
for step in range(merge_count):
    stats = {}
    for chunk_ids in tokens:
        get_stats(chunk_ids, stats)

    most_common = max(stats, key=stats.get)
    step_idx = step + 256
    tokens = [merge(chunk_ids, most_common, step_idx) for chunk_ids in tokens]
    merges[most_common] = step_idx
    vocab[step_idx] = vocab[most_common[0]] + vocab[most_common[1]]


print("Resulting vocab:")
for idx, rep in vocab.items():
    print(f"{idx} => {rep.decode(errors='replace')}")

with open("data/merges.pkl", "wb") as file:
    pickle.dump(merges, file)


with open("data/vocab.pkl", "wb") as file:
    pickle.dump(vocab, file)
