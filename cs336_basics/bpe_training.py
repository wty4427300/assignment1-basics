from __future__ import annotations

import collections
import regex
from concurrent.futures import ProcessPoolExecutor

# This pattern requires the `regex` module and is compiled here for efficiency.
import os
from typing import BinaryIO


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

    # 2. Read and chunk by special tokens
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    special_tokens_set = set(special_tokens)
    if special_tokens:
        special_pattern = f"({'|'.join(regex.escape(st) for st in special_tokens)})"
        chunks = regex.split(special_pattern, text)
    else:
        chunks = [text]

    # 3. Parallel pre-tokenization of text chunks
    text_chunks = [chunk for chunk in chunks if chunk and chunk not in special_tokens_set]
    word_counts = collections.Counter()
    with ProcessPoolExecutor() as executor:
        word_lists = executor.map(_tokenize_chunk, text_chunks)
        for word_list in word_lists:
            word_counts.update(word_list)

    # Add back special tokens to word counts
    for chunk in chunks:
        if chunk in special_tokens_set:
            word_counts[chunk] += 1

    # 4. Initialize splits
    splits = collections.Counter()
    for word_str, freq in word_counts.items():
        if word_str in special_tokens_set:
            continue
        splits[tuple(bytes([b]) for b in word_str.encode("utf-8"))] = freq

    # 5. Main merge loop (optimized)
    merges = []
    num_merges = vocab_size - len(vocab)
    pair_stats = get_pair_stats(splits)

    for i in range(num_merges):
        if not pair_stats:
            break
        
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        if pair_stats[best_pair] < 1:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
        # In-place merge and stats update
        p1, p2 = best_pair
        merged_token = p1 + p2
        
        # Find words affected by the merge
        words_to_update = []
        for word in splits:
            if len(word) < 2:
                continue
            for j in range(len(word) - 1):
                if word[j] == p1 and word[j+1] == p2:
                    words_to_update.append(word)
                    break

        for word_to_update in words_to_update:
            if word_to_update not in splits:
                continue
            
            freq = splits.pop(word_to_update)
            
            if len(word_to_update) >= 2:
                for p in zip(word_to_update, word_to_update[1:]):
                    pair_stats[p] -= freq

            new_word = []
            j = 0
            while j < len(word_to_update):
                if j < len(word_to_update) - 1 and word_to_update[j] == p1 and word_to_update[j+1] == p2:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(word_to_update[j])
                    j += 1
            new_word = tuple(new_word)
            splits[new_word] += freq

            if len(new_word) >= 2:
                for p in zip(new_word, new_word[1:]):
                    pair_stats[p] += freq

        del pair_stats[best_pair]

    return vocab, merges

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
    Trains a BPE tokenizer from a text file.
    """
    # 1. Initialize vocab with base tokens and special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # 2. Correctly pre-tokenize the text by splitting by special tokens first
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    all_words = []
    special_tokens_set = set(special_tokens)
    if special_tokens:
        special_pattern = f"({'|'.join(regex.escape(st) for st in special_tokens)})"
        chunks = regex.split(special_pattern, text)
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            if i % 2 == 1:  # This chunk is a special token
                all_words.append(chunk)
            else:  # This is a regular text chunk
                all_words.extend(regex.findall(GPT2_SPLIT_PATTERN, chunk))
    else:
        all_words.extend(regex.findall(GPT2_SPLIT_PATTERN, text))
    
    word_counts = collections.Counter(all_words)

    # 3. Initialize splits, excluding special tokens from the merge process
    splits = collections.Counter()
    for word_str, freq in word_counts.items():
        if word_str in special_tokens_set:
            continue
        word_bytes = word_str.encode("utf-8")
        splits[tuple(bytes([b]) for b in word_bytes)] = freq

    # 4. Main merge loop with tie-breaking and efficient in-place merge
    merges = []
    num_merges = vocab_size - len(vocab)

    for i in range(num_merges):
        pair_stats = get_pair_stats(splits)
        if not pair_stats:
            break
        
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merge_pair(splits, best_pair)  # In-place modification

    return vocab, merges


def get_pair_stats(splits: dict[tuple[bytes, ...], int]) -> collections.Counter:
    """Counts frequencies of adjacent pairs in all word splits."""
    stats = collections.Counter()
    for word_tokens, freq in splits.items():
        if len(word_tokens) < 2:
            continue
        for pair in zip(word_tokens, word_tokens[1:]):
            stats[pair] += freq
    return stats


def merge_pair(splits: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]):
    """Merges a pair of bytes in all applicable word splits (in-place)."""
    p1, p2 = pair
    merged_token = p1 + p2
    
    words_to_change = []
    for word_tokens in splits:
        i = 0
        while i < len(word_tokens) - 1:
            if word_tokens[i] == p1 and word_tokens[i+1] == p2:
                words_to_change.append(word_tokens)
                break
            i += 1

    for word_tokens in words_to_change:
        freq = splits.pop(word_tokens)
        
        new_word_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and word_tokens[i] == p1 and word_tokens[i+1] == p2:
                new_word_tokens.append(merged_token)
                i += 2
            else:
                new_word_tokens.append(word_tokens[i])
                i += 1
        
        splits[tuple(new_word_tokens)] += freq
