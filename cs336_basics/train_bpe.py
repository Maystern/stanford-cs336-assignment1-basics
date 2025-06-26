from typing import Tuple, Dict, List
from pretokenization import find_chunk_boundaries, find_special_token_pos

import os
import multiprocessing as mp
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def tokenization(chunk: bytes, special_tokens: list[str]) -> dict[tuple[bytes], int]:

    found_at, length = find_special_token_pos(chunk, special_tokens)
    special_token = b""
    text = chunk

    if found_at != -1:
        special_token = chunk[: found_at + length]
        text = chunk[found_at + length:]
            
    # matches_list = list(re.finditer(PAT, text))
    # return [match.group() for match in matches_list]

    matches_list = text.split(" ")

    word_count= {}
    for match in matches_list:
        tuple_mem = tuple([b for b in match])
        if tuple_mem in word_count:
            word_count[tuple_mem] += 1
        else:
            word_count[tuple_mem] = 1
    return word_count



def train_bpe (
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    """
    返回值 Dict[int, bytes] 表示 int -> 字符子串，初始时 utf-8 的 256 个值（+特殊的token），后续需要扩充打 vocab_size 大小的词表，首先需要确认 vocab_size >= 256
    返回值 List[tuple[bytes, bytes]] 表示 merge 的相邻的两个 bytes 数组的值
    """

    assert vocab_size >= len(special_tokens) + 256, "train_bpe err: vocab_size is too small"

    vocab = {}

    vocab_idx = 0
    for vocab_idx in range(256):
        vocab[vocab_idx] = bytes([vocab_idx])
    
    
    for special_token in special_tokens:
        vocab_idx += 1
        vocab[vocab_idx] = special_token
        
    num_epoches = vocab_size - (len(special_tokens) + 256)

    global_special_tokens = special_tokens

    num_processes = os.cpu_count()
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]

    chunks = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_utf8)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk, special_tokens))

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(tokenization, chunks)
    
    #todo: 优化点：暴力 merge，得到一个大 dict[tuple[bytes], int]
    token_count = {}
    for result in results:
        for token, count in result.items():
            if token in token_count:
                token_count[token] += count
            else:
                token_count[token] = count
    
    adj_token_count = {}
    for token, count in token_count.items():
        for t1, t2 in zip(token[:-1], token[1:]):
            t = (t1, t2)
            if t in adj_token_count:
                adj_token_count[t] += count
            else:
                adj_token_count[t] = count

    print(token_count)
    print(adj_token_count)

    merge_list = []

    for epoch in num_epoches:
        # merge
        # step1 找到出现最大次数的 adj_token, 次数一致的情况下选择字典序最大的
        max_adj_token = None
        max_count = 0
        for adj_token, count in adj_token_count.items():
            if count > max_count:
                max_count = count
                max_adj_token = adj_token
            elif count == max_count and (max_adj_token == None or max_adj_token[0] + max_adj_token[1] < adj_token[0] + adj_token[1]):
                max_adj_token = adj_token
        
        assert max_adj_token is None, "train_bpe err: vocab_size is too large"
        
        # step2 更新 vocab 表
        vocab_idx += 1
        vocab[vocab_idx] = max_adj_token
        merge_list.append(max_adj_token)

        # step3 更新两个map
        



if __name__ == "__main__":
    # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "../data/hello.txt"
    vocab_size = 256 + 2
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)