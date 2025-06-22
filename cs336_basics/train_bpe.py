from typing import Tuple, Dict, List
from pretokenization import find_chunk_boundaries
import os

def train_bpe (
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    """
    返回值 Dict[int, bytes] 表示 int -> 字符子串，初始时 utf-8 的 256 个值，后续需要扩充打 vocab_size 大小的词表，首先需要确认 vocab_size >= 256
    返回值 List[tuple[bytes, bytes]] 表示 merge 的相邻的两个 bytes 数组的值
    """
    num_processes = os.cpu_count()
    special_tokens_utf8 = [special_token.encode("utf-8") for special_token in special_tokens]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_utf8)
        print(boundaries)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
    


if __name__ == "__main__":
    # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "../data/hello.txt"
    vocab_size = 256 
    special_tokens = ["<|endoftext|>", "\r", "\n"]
    train_bpe(input_path, vocab_size, special_tokens)