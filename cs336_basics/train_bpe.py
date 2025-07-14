from typing import Tuple, Dict, List, Set
from tqdm import tqdm
from cs336_basics.pretokenization import find_chunk_boundaries, find_all_regex
from cs336_basics.consts import PAT

import os
import multiprocessing as mp
import regex as re
import pickle
import time



def init_pool(lock):
    global tqdm_lock
    tqdm_lock = lock

def split_by_special_tokens(chunk: bytes | str, special_tokens:list[str]) -> list[bytes]:
    special_token_ranges = []
    for special_token in special_tokens:
        left_bounds = find_all_regex(chunk, special_token)
        for left_bound in left_bounds:
            special_token_ranges.append((left_bound, left_bound + len(special_token)))
    special_token_ranges = sorted(special_token_ranges)
    
    # 为了保证表示字符串的 token 尽可能短，因此某个点位选择的 special tokens 要进可能长
    # 排序过后，选择相同 left 的最后一个 right

    flitered_special_token_ranges = []
    i = 0
    while i < len(special_token_ranges):
        j = i
        while j < len(special_token_ranges) and special_token_ranges[i][0] == special_token_ranges[j][0]:
            j += 1
        j -= 1
        flitered_special_token_ranges.append(special_token_ranges[j])
        i = j + 1
    
    special_token_ranges = flitered_special_token_ranges
    splited_chunk = []
    now_pos = 0
    for lb, rb in special_token_ranges:
        # [: now_pos) 已经完成分割，现在来了一个[lb, rb)
        # 合法的只有 lb >= now_pos 才合法
        if lb < now_pos:
            continue
        splited_chunk.append(chunk[now_pos: lb])
        splited_chunk.append(chunk[lb: rb])
        now_pos = rb
    if now_pos != len(chunk):
        splited_chunk.append(chunk[now_pos: ])
    return splited_chunk

def tokenization(args) -> dict[tuple[bytes], int]:
    # task_id: int, chunk: bytes, special_tokens: list[str]
    task_id, chunk, special_tokens = args
    word_count= {}
    splited_chunk = split_by_special_tokens(chunk, special_tokens)
    special_tokens_set = set(special_tokens)

    for text in splited_chunk:
        if text in special_tokens_set:
            continue
        # 使用 gpt2 tokenization
        matches_list = list(re.finditer(PAT, text))
        matches_list = [match.group().encode("utf-8") for match in matches_list]

        # 只使用 split(' ') 进行 tokenization
        # matches_list = list(text.split(" "))
        # matches_list = [match.encode("utf-8") for match in matches_list]
        for match in matches_list:
            tuple_mem = tuple([bytes([b]) for b in match])
            if tuple_mem in word_count:
                word_count[tuple_mem] += 1
            else:
                word_count[tuple_mem] = 1

    return word_count


class SingleToken:
    
    token: Tuple[bytes] # 完整的 token 可能由几部分组成，如最开始是 (b'h', b'e', b'l', b'l', b'o') 经过几轮合并后可能为 (b'hel', b'lo')
    adj_token_set: Set[Tuple[bytes, bytes]] # byte pairs 的缓存，如 (b'h', b'e') 在内部，而 (b'a', b'b') 不在
    wordCnt: int # token 在 文字中出现的次数
    adj_token_count: Dict[Tuple[bytes], int] # 全局的 byte pairs -> 出现次数的映射
    
    def __init__(self, token: Tuple[bytes], wordCnt: int, adj_token_count: Dict[Tuple[bytes], int]):
        self.token = token
        self.adj_token_set = set()
        self.adj_token_count = adj_token_count
        for t1, t2 in zip(token[: -1], token[1:]):
            key = (t1, t2)
            self.adj_token_set.add(key)
            adj_token_count[key] = adj_token_count.get(key, 0) + wordCnt
        self.wordCnt = wordCnt
    
    def merge(self, pair: Tuple[bytes, bytes]):
        # 增加一层 cache，如果当前 byte pair 不在这个 token 里的话，直接 return
        if pair not in self.adj_token_set:
            return
        
        # 扣减掉原来的贡献
        for t1, t2 in zip(self.token[: -1], self.token[1:]):
            key = (t1, t2)
            self.adj_token_count[key] -= self.wordCnt

        # 合并，注意 a,a,a,a 合并 aa 后将成为 aa,aa
        next_token = []
        i = 0
        while i < len(self.token):
            if i + 1 < len(self.token) and (self.token[i], self.token[i + 1]) == pair:
                next_token.append(self.token[i] + self.token[i + 1])
                i = i + 1
            else:
                next_token.append(self.token[i])
            i = i + 1
        next_token = tuple(next_token)
        
        # 加上新增的贡献
        self.token = next_token
        self.adj_token_set = set()
        for t1, t2 in zip(self.token[: -1], self.token[1:]):
            key = (t1, t2)
            self.adj_token_set.add(key)
            self.adj_token_count[key] = self.adj_token_count.get(key, 0) + self.wordCnt
        

def train_bpe (
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    """
    返回值 Dict[int, bytes] 表示 int -> 字符子串，初始时 utf-8 的 256 个值（+特殊的token），后续需要扩充打 vocab_size 大小的词表，首先需要确认 vocab_size >= 256
    返回值 List[tuple[bytes, bytes]] 表示 merge 的相邻的两个 bytes 数组的值
    """
    start_time = time.time()
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]
    
    vocab = {}
    vocab_idx = 0

    # 提前将 utf-8 的 256 个编码写入词表
    for vocab_idx in range(256):
        vocab[vocab_idx] = bytes([vocab_idx])
    
    # 提前将 special tokens 写入词表
    for special_token in special_tokens_utf8:
        vocab_idx += 1
        vocab[vocab_idx] = special_token
    
    # 计算还需要 merge 多少轮，才能使得词表大小为 vocab_size
    num_epoches = vocab_size - vocab_idx - 1

    # 计算当前设备 cpu 数量，分块进行 pre-tokenization
    num_processes = os.cpu_count()
    print(f"[{(time.time() - start_time):.2f}s] loading dataset using {num_processes} cpus.")
    params = []
    task_id = 0
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_utf8)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            params.append((task_id, chunk, special_tokens))
            task_id += 1
    
    print(f"[{(time.time() - start_time):.2f}s] all chunks: {[len(p[1]) for p in params]}")


    with mp.Manager() as manager:
        tqdm_lock = manager.Lock()
        with mp.Pool(len(params), initializer=init_pool, initargs=(tqdm_lock,)) as pool:
            results = pool.map(tokenization, params)

    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.starmap(tokenization, params)
    
    
    print(f"[{(time.time() - start_time):.2f}s] merging results from all cpus.")
    # 将多 cpu 计算的结果合并
    token_count = {}
    for result in results:
        for token, count in result.items():
            if token in token_count:
                token_count[token] += count
            else:
                token_count[token] = count
    
    print(f"[{(time.time() - start_time):.2f}s] building singletoken.")
    adj_token_count = {}
    token_count_list = []
    for token, count in token_count.items():
        token_count_list.append(SingleToken(token, count, adj_token_count))
    
    merge_list = []

    for _ in tqdm(range(num_epoches), desc="Training BPE"):
        # 找到出现最大次数的 adj_token, 次数一致的情况下选择字典序最大的
        max_adj_token = None
        max_count = 0
        for adj_token, count in adj_token_count.items():
            if count > max_count:
                max_count = count
                max_adj_token = adj_token
            elif count == max_count and ((max_adj_token == None) or max_adj_token < adj_token):
                max_adj_token = adj_token
        
        assert max_adj_token is not None, "train_bpe err: vocab_size is too large"
        
        # 更新 vocab 表
        vocab_idx += 1
        vocab[vocab_idx] = max_adj_token[0] + max_adj_token[1]
        merge_list.append(max_adj_token)

        # 更新每个 token 的 cache 和 adj_token 
        for token_count in token_count_list:
            token_count.merge(max_adj_token)
    
    print(f"[{(time.time() - start_time):.2f}s] done.")

    return vocab, merge_list

if __name__ == "__main__":
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/owt_valid.txt"
    dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/owt_train.txt"
    store_path_base = "/Users/bytedance/code/stanford-cs336-assignment1-basics/results/"
    dataset_name = store_path_base + dataset_path.split("/")[-1].split(".")[0]

    special_tokens = ["<|endoftext|>"]
    vocab_size = 10_000
    vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens)

    with open(dataset_name + "_vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(dataset_name + "_merges.pkl", 'wb') as f:
        pickle.dump(merges, f)

    # with open(dataset_name + "_vocab.pkl", 'rb') as f:
    #     loaded_data = pickle.load(f)
    #     print(loaded_data)

    # with open(dataset_name + "_merges.pkl", 'rb') as f:
    #     loaded_data = pickle.load(f)
    #     print(loaded_data)