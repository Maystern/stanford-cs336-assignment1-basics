from typing import BinaryIO, List, Tuple
import os
import re


def find_all_regex(data: bytes | str, sub: bytes | str) -> List[int]:
    """使用正则表达式查找子字节串的所有位置"""
    # 注意：需要使用 re.escape() 处理子字节串中的特殊字符
    pattern = re.escape(sub)
    return [m.start() for m in re.finditer(pattern, data)]


def find_special_token_pos(pattern: bytes, special_tokens: List[bytes]) -> Tuple[int, int]:
    found_at = -1
    length = -1
    for special_token in special_tokens:
        far = find_all_regex(pattern, special_token)
        if len(far) == 0:
            found_at_this = -1
        else:
            found_at_this = min(far)
        if found_at_this != -1:
            if found_at == -1:
                found_at = found_at_this
                length = len(special_token)
            elif (found_at_this < found_at):
                found_at = found_at_this
                length = len(special_token)
    return found_at, length

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    special_tokens: List[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    for special_token in special_tokens:
        assert isinstance(special_token, bytes), (
            "Must represent special token as a bytestring"
        )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at, _ = find_special_token_pos(mini_chunk, special_tokens)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
if __name__ == "__main__":
    with open(..., "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token