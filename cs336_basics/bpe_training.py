from __future__ import annotations

import collections
import regex
from concurrent.futures import ProcessPoolExecutor

# This pattern requires the `regex` module and is compiled here for efficiency.
GPT2_SPLIT_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _tokenize_chunk(chunk: str) -> list[str]:
    """
    A picklable, top-level function to be used by ProcessPoolExecutor.
    Takes a string chunk and returns a list of pre-tokenized words.
    """
    return regex.findall(GPT2_SPLIT_PATTERN, chunk)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer from a text file using parallel pre-tokenization.
    """
    # 1. Initialize Vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # 2. Read and chunk by special tokens (this part is fast)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_pattern = "|".join(regex.escape(token) for token in special_tokens)
        chunks = regex.split(f"({special_pattern})", text)
    else:
        chunks = [text]

    # 3. Parallel pre-tokenization of text chunks
    special_tokens_set = set(special_tokens)
    text_chunks = [chunk for chunk in chunks if chunk and chunk not in special_tokens_set]

    with ProcessPoolExecutor() as executor:
        # Process all text chunks in parallel
        word_lists = list(executor.map(_tokenize_chunk, text_chunks))

    # Create a map from the original text chunk to its tokenized version
    chunk_to_words = dict(zip(text_chunks, word_lists))

    # 4. Reconstruct the full list of words in the correct order
    all_words = []
    for chunk in chunks:
        if not chunk:
            continue
        if chunk in special_tokens_set:
            all_words.append(chunk)
        else:
            all_words.extend(chunk_to_words[chunk])

    # 5. Count word frequencies
    word_counts = collections.Counter(all_words)

    # 6. Convert word counts to a list of byte sequences for merging
    splits = collections.Counter()
    for word_str, freq in word_counts.items():
        # Encode the string word into a tuple of its bytes
        splits[tuple(word_str.encode("utf-8"))] = freq

    # 7. Main merge loop
    merges = []
    num_merges = vocab_size - len(vocab)

    for i in range(num_merges):
        # a. Count pair frequencies
        pair_stats = get_pair_stats(splits)

        # b. Find the most frequent pair
        if not pair_stats:
            break  # No more pairs to merge

        best_pair = max(pair_stats, key=pair_stats.get)

        # c. Merge the best pair
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        splits = merge_pair(splits, best_pair)

    # Return the trained vocab and merges
    return vocab, merges


def get_pair_stats(splits: dict[tuple[bytes, ...], int]) -> collections.Counter:
    """Counts frequencies of adjacent pairs in all word splits."""
    stats = collections.Counter()
    for word_bytes, freq in splits.items():
        # Use zip to efficiently create pairs of adjacent items
        for pair in zip(word_bytes, word_bytes[1:]):
            stats[pair] += freq
    return stats


def merge_pair(
    splits: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    """Merges a pair of bytes in all word splits."""
    new_splits = collections.Counter()
    p1, p2 = pair
    merged_token = p1 + p2
    for word_bytes, freq in splits.items():
        new_word = []
        i = 0
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and word_bytes[i] == p1 and word_bytes[i+1] == p2:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word_bytes[i])
                i += 1
        new_splits[tuple(new_word)] += freq
    return new_splits
