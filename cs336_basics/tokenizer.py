from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import regex as re


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """返回 GPT-2 使用的 byte-to-unicode 映射。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


class Tokenizer:
    """
    一个 BPE 分词器，能够将文本编码为 token ID 序列，以及将 ID 序列解码回文本。
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                 special_tokens: Optional[List[str]] = None):
        """
        从给定的词汇表、合并规则和特殊词元列表构造一个分词器。

        参数:
            vocab (Dict[int, bytes]): 从 token ID (int) 到 token (bytes) 的映射。
            merges (List[Tuple[bytes, bytes]]): BPE 合并规则的有序列表。
            special_tokens (Optional[List[str]]): 特殊词元的字符串列表，例如 ["<|endoftext|>"].
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}
        self.special_tokens_set = set(self.special_tokens)

        if self.special_tokens:
            # 按长度降序构造 alternation，避免重叠 special token 被较短前缀抢先匹配。
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(map(re.escape, sorted_special_tokens))
            self.special_regex = re.compile(f"({special_pattern})")
            for token_str in self.special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.inverse_vocab[token_bytes] = new_id
        else:
            self.special_regex = None

        self.pretokenize_regex = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: Optional[List[str]] = None) -> "Tokenizer":
        """
        一个类方法，从文件加载词汇表和合并规则来构造分词器实例。
        """
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
        vocab = {
            token_id: bytes(byte_decoder[ch] for ch in token_str)
            for token_str, token_id in raw_vocab.items()
        }

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip().split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in left),
                        bytes(byte_decoder[ch] for ch in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _encode_piece(self, piece: str) -> List[int]:
        """将一个非 special-token 的预分词片段编码成 token IDs。"""
        if not piece:
            return []

        tokens = [bytes([b]) for b in piece.encode("utf-8")]
        while len(tokens) > 1:
            best_index = None
            best_rank = None
            for i, pair in enumerate(zip(tokens, tokens[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break

            tokens = (
                tokens[:best_index]
                + [tokens[best_index] + tokens[best_index + 1]]
                + tokens[best_index + 2:]
            )

        return [self.inverse_vocab[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        将输入的文本字符串编码成一个 token ID 列表。
        """
        ids: List[int] = []

        if self.special_regex is None:
            pieces = [text]
        else:
            pieces = self.special_regex.split(text)

        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens_set:
                ids.append(self.inverse_vocab[piece.encode("utf-8")])
                continue
            for word in self.pretokenize_regex.findall(piece):
                ids.extend(self._encode_piece(word))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个字符串的可迭代对象进行编码，并惰性地产生 token ID。
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: List[int]) -> str:
        """
        将一个 token ID 列表解码回文本字符串。
        """
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
