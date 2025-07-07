from typing import Iterable, Iterator, List, Dict, Tuple, Set
from cs336_basics.consts import PAT
from cs336_basics.train_bpe import split_by_special_tokens

import pickle
import regex as re

class Tokenizer:
    special_tokens: List[str] | None = None
    token_to_code: Dict[Tuple[bytes], List[int]]
    special_tokens_set: Set[str]

    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        if special_tokens is None:
            special_tokens = []
        self.vocab = vocab
        self.rev_vocab = {}
        for code, token in vocab.items():
            self.rev_vocab[token] = code        
        self.merges_dict = {}
        for id, merge in enumerate(merges):
            self.merges_dict[merge] = id
        self.special_tokens = special_tokens
        self.token_to_code = {}
        self.special_tokens_set = set(special_tokens)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens:list[str] | None = None) -> 'Tokenizer':
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encode_result = []
        chunks = split_by_special_tokens(text, self.special_tokens)
        for text in chunks:
            if text in self.special_tokens_set:
                encode_result.append(self.rev_vocab[text.encode("utf-8")])
                continue
            matches_list = list(re.finditer(PAT, text))
            matches_list = [match.group().encode("utf-8") for match in matches_list]
            for match in matches_list:
                token = tuple([bytes([b]) for b in match])
                if token in self.token_to_code:
                    encode_result.extend(self.token_to_code[token])
                    continue
                rec_token = tuple(token)
                while True:
                    min_merge_idx = -1
                    min_merge = (None, None)
                    min_merge_pos = -1
                    pos = 0
                    for t1, t2 in zip(token[: -1], token[1: ]):
                        key = (t1, t2)
                        merge_idx = self.merges_dict.get(key, -1)
                        if merge_idx != -1 and (min_merge_idx == -1 or merge_idx < min_merge_idx):
                            min_merge_idx = merge_idx
                            min_merge = key
                            min_merge_pos = pos
                        pos += 1

                    if min_merge_idx == -1:
                        break

                    next_token = list(token[: min_merge_pos])
                    next_token.append(b"".join(min_merge))
                    next_token.extend(token[min_merge_pos + 2:])
                    
                    token = tuple(next_token)

                rec_code = []
                for pair in token:
                    rec_code.append(self.rev_vocab[pair])
                self.token_to_code[rec_token] = rec_code
                encode_result.extend(rec_code)
        return encode_result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_list = [self.vocab[id] for id in ids]
        return b"".join(token_list).decode("utf-8", errors="replace")

if __name__ == "__main__":
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/owt_valid.txt"
    # dataset_path = "/Users/bytedance/code/stanford-cs336-assignment1-basics/data/owt_train.txt"
    store_path_base = "/Users/bytedance/code/stanford-cs336-assignment1-basics/results/"
    dataset_name = store_path_base + dataset_path.split("/")[-1].split(".")[0]

    vocab_path = dataset_name + "_vocab.pkl"
    merges_path = dataset_name + "_merges.pkl"
    special_tokens = ["<|endoftext|>"]


    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(merges_path + "", 'rb') as f:
        merges = pickle.load(f)
    
    tokenizer1 = Tokenizer(vocab, merges, special_tokens)
    encode = tokenizer1.encode("👋<|endoftext|>中国")
    print(f"encode = {encode}")
    decode = tokenizer1.decode(encode)
    print(f"decode = {decode}")
    # tokenizer2 = Tokenizer.from_files(vocab_path, merges_path, special_tokens)