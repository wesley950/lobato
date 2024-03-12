import pickle
import regex as re

from .common import get_stats, merge, pattern

with open("data/merges.pkl", "rb") as file:
    merges = pickle.load(file)

with open("data/vocab.pkl", "rb") as file:
    vocab = pickle.load(file)


def _encode_chunk(chunk_bytes: bytes):
    tokens = list(chunk_bytes)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


def encode(text: str):
    chunks = re.findall(pattern, text, flags=re.IGNORECASE)
    tokens = []

    for chunk in chunks:
        chunk_bytes = chunk.encode()
        chunk_tokens = _encode_chunk(chunk_bytes)
        tokens.extend(chunk_tokens)

    return tokens


def decode(tokens: list[int]):
    items = b"".join(vocab[idx] for idx in tokens)
    return items.decode(errors="replace")


print(f"Vocab size: {len(vocab)}")
