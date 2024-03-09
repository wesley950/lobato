def get_most_common_byte_pair(tokens: list[int]) -> dict:
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


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
