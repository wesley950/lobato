import torch
import regex as re

pattern = r" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|s+"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_stats(tokens: list[int], current_stats: dict | None = None) -> dict:
    if current_stats == None:
        current_stats = {}
    for pair in zip(tokens, tokens[1:]):
        current_stats[pair] = current_stats.get(pair, 0) + 1
    return current_stats


def merge(ids, pair, idx):
    new_ids = []
    step = 0
    while step < len(ids):
        if step + 1 < len(ids) and pair[0] == ids[step] and pair[1] == ids[step + 1]:
            new_ids.append(idx)
            step += 2
        else:
            new_ids.append(ids[step])
            step += 1
    return new_ids
