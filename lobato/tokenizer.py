import pickle
from .common import get_most_common_byte_pair, merge

with open("data/merges.pkl", "rb") as file:
    merges = pickle.load(file)

with open("data/vocab.pkl", "rb") as file:
    vocab = pickle.load(file)


def encode(text: str):
    text_tokens = text.encode()
    text_tokens = list(map(int, text_tokens))
    while True:
        stats = get_most_common_byte_pair(text_tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        text_tokens = merge(text_tokens, pair, idx)

    return text_tokens


def decode(tokens: list[int]):
    items = b"".join(vocab[idx] for idx in tokens)
    return items.decode(errors="replace")


msg = "O Campus Darcy Ribeiro é o campus mais antigo da Universidade de Brasília."
print(encode(msg))
print(msg)
print(decode(encode(msg)))
