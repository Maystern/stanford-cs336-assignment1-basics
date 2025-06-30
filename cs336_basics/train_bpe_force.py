from typing import Tuple, Dict, List
from cs336_basics.pretokenization import find_chunk_boundaries, find_special_token_pos
from tqdm import tqdm

import os
import multiprocessing as mp
import regex as re
import cProfile


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def tokenization(chunk: bytes, special_tokens: list[str]) -> dict[tuple[bytes], int]:

    now_idx = 0
    word_count= {}

    while now_idx < len(chunk):
        found_at, length = find_special_token_pos(chunk[now_idx:], special_tokens)
        if found_at == -1:
            text = chunk[now_idx:]
            now_idx = len(chunk)
        else:
            text = chunk[now_idx: now_idx + found_at]
            now_idx = now_idx + found_at + length
        matches_list = list(re.finditer(PAT, text))
        matches_list = [match.group().encode("utf-8") for match in matches_list]
        for match in matches_list:
            tuple_mem = tuple([bytes([b]) for b in match])
            if tuple_mem in word_count:
                word_count[tuple_mem] += 1
            else:
                word_count[tuple_mem] = 1
    
    return word_count

class SingleToken:
    token: Tuple[bytes]
    wordCnt: int
    def __init__(self, token: Tuple[bytes], count: int, adj_token_count: Dict[Tuple[bytes], int]):
        self.token = token
        self.wordCnt = count
        for t1, t2 in zip(token[:-1], token[1:]):
            t = (t1, t2)
            if t in adj_token_count:
                adj_token_count[t] += count
            else:
                adj_token_count[t] = count


def train_bpe (
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    """
    返回值 Dict[int, bytes] 表示 int -> 字符子串，初始时 utf-8 的 256 个值（+特殊的token），后续需要扩充打 vocab_size 大小的词表，首先需要确认 vocab_size >= 256
    返回值 List[tuple[bytes, bytes]] 表示 merge 的相邻的两个 bytes 数组的值
    """
    profiler = cProfile.Profile()
    profiler.enable()

    assert vocab_size >= len(special_tokens) + 256, "train_bpe err: vocab_size is too small"
    
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]
    
    vocab = {}
    vocab_idx = 0

    for vocab_idx in range(256):
        vocab[vocab_idx] = bytes([vocab_idx])
    
    # 词表中 special tokens 需要提前写入
    for special_token in special_tokens_utf8:
        vocab_idx += 1
        vocab[vocab_idx] = special_token
        
    num_epoches = vocab_size - 256 - len(special_tokens)

    num_processes = os.cpu_count()

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
    # for token, count in token_count.items():
    #     for t1, t2 in zip(token[:-1], token[1:]):
    #         t = (t1, t2)
    #         if t in adj_token_count:
    #             adj_token_count[t] += count
    #         else:
    #             adj_token_count[t] = count

    single_token_list = []
    for token, count in token_count.items():
        single_token_list.append(SingleToken(token, count, adj_token_count))

    merge_list = []

    for epoch in tqdm(range(num_epoches)):
        # merge
        # step1 找到出现最大次数的 adj_token, 次数一致的情况下选择字典序最大的
        max_adj_token = None
        max_count = 0
        for adj_token, count in adj_token_count.items():
            if count > max_count:
                max_count = count
                max_adj_token = adj_token
            elif count == max_count and ((max_adj_token == None) or max_adj_token < adj_token):
                max_adj_token = adj_token
        
        assert max_adj_token is not None, "train_bpe err: vocab_size is too large"
        
        # step2 更新 vocab 表
        vocab_idx += 1
        vocab[vocab_idx] = max_adj_token[0] + max_adj_token[1]
        merge_list.append(max_adj_token)

        # step3 更新两个map
        next_token_count = {}

        for token, count in token_count.items():
            # 扣减掉原来的贡献
            for t1, t2 in zip(token[:-1], token[1:]):
                t = (t1, t2)
                adj_token_count[t] -= count
                if adj_token_count[t] == 0:
                    del adj_token_count[t]

            # 合并
            next_token = []
            i = 0
            while i < len(token):
                if i + 1 < len(token) and (token[i], token[i + 1]) == max_adj_token:
                    next_token.append(token[i] + token[i + 1])
                    i = i + 1
                else:
                    next_token.append(token[i])
                i = i + 1

            # 加上新增的贡献
            for t1, t2 in zip(next_token[: -1], next_token[1:]):
                t = (t1, t2)
                if t in adj_token_count:
                    adj_token_count[t] += count
                else:
                    adj_token_count[t] = count

            next_token = tuple(next_token)
            if next_token in next_token_count:
                next_token_count[next_token] += count
            else:
                next_token_count[next_token] = count

        token_count = next_token_count
    
    profiler.disable()
    profiler.dump_stats("profile_results.prof")  # 保存分析结果
    return vocab, merge_list



if __name__ == "__main__":
    # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/hello.txt"
    # input_path = "../data/bb.txt"
    vocab_size = 256 + 10
    special_tokens = ["<|endoftext|>"]
    vocab, merge_list = train_bpe(input_path, vocab_size, special_tokens)
    print(vocab)
    print(merge_list)