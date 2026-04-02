from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor

import regex


# GPT-2 风格的预分词正则；BPE 合并发生在字节级别，因此先按这个规则切成词片段。
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
        # 这里使用捕获组，保证 special token 在 split 结果里被完整保留下来，
        # 这样既能统计它们出现的次数，又不会把它们送进后续的 BPE merge 过程。
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
    # splits 的键是“一个预分词后的词”对应的字节 token 序列，
    # 值是这个字节 token 序列在语料中出现的频次。
    splits = collections.Counter()
    for word_str, freq in word_counts.items():
        if word_str in special_tokens_set:
            continue
        splits[tuple(bytes([b]) for b in word_str.encode("utf-8"))] = freq

    # 5. Main merge loop (optimized)
    merges = []
    num_merges = vocab_size - len(vocab)
    # pair_stats 先整体统计一次；之后每轮 merge 只增量更新受影响的词，避免反复全量重算。
    pair_stats = get_pair_stats(splits)

    for i in range(num_merges):
        if not pair_stats:
            break
        
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        if pair_stats[best_pair] < 1:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
        # 应用本轮最优 merge，并只更新受影响词的 pair 统计；
        # 这样比每一轮都重新扫描整个语料更快。
        p1, p2 = best_pair
        merged_token = p1 + p2
        
        # 只有包含当前 best_pair 的词才会在这一轮发生变化。
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
            
            # 先把旧词形对应的相邻 pair 贡献从统计里减掉。
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

            # 再把新词形对应的相邻 pair 贡献加回去。
            if len(new_word) >= 2:
                for p in zip(new_word, new_word[1:]):
                    pair_stats[p] += freq

        # 当前选中的 pair 已经被合并，不再作为下一轮候选。
        del pair_stats[best_pair]

    return vocab, merges


def get_pair_stats(splits: dict[tuple[bytes, ...], int]) -> collections.Counter:
    """统计整个语料里相邻字节 token 对的频次。"""
    stats = collections.Counter()
    for word_tokens, freq in splits.items():
        if len(word_tokens) < 2:
            continue
        for pair in zip(word_tokens, word_tokens[1:]):
            stats[pair] += freq
    return stats


def merge_pair(splits: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]):
    """
    在所有包含给定 pair 的词上原地执行 merge。

    这个函数是“全量重写受影响词”的直接实现，逻辑上更直观，
    但当前 train_bpe 主流程走的是增量更新 pair_stats 的优化路径，没有直接调用它。
    """
    p1, p2 = pair
    merged_token = p1 + p2

    # 先找出所有包含目标 pair 的词，避免一边遍历一边修改 Counter。
    
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

        # 把词里的所有目标 pair 都替换成 merge 后的新 token。
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
