from typing import Dict, List, Tuple, Optional, Iterable, Iterator
import regex as re


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
            special_tokens (Optional[List[str]]): 特殊词元的字符串列表，例如 ["<|endoftext|>"]。
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 1. 反向词汇表
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        # 2. 合并规则
        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}
        # 3. 处理特殊词元，将它们添加到词汇表中。
        self.special_tokens_set = set(self.special_tokens)
        if self.special_tokens:
            # 如果有特殊词元，我们需要构建一个正则表达式来分割它们
            # 例如，如果 special_tokens 是 ["<|endoftext|>"]
            # 这个正则表达式就会变成 "(<|endoftext|>)"
            # re.escape() 是为了防止特殊词元中包含正则表达式的特殊字符
            special_pattern = "|".join(map(re.escape, self.special_tokens))
            self.special_regex = re.compile(f"({special_pattern})")
            # 将特殊词元也加入到词汇表中
            for token_str in self.special_tokens:
                if token_str not in self.inverse_vocab:
                    # 找到一个还没被使用的ID
                    new_id = len(self.vocab)
                    # 将特殊词元编码为 bytes
                    token_bytes = token_str.encode("utf-8")
                    # 更新两个词汇表
                    self.vocab[new_id] = token_bytes
                    self.inverse_vocab[token_bytes] = new_id
        else:
            self.special_regex = None
            # 4. GPT-2使用的核心预分词正则表达式
            # 这个复杂的表达式能够很好地处理各种文本情况
        self.pretokenize_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: Optional[List[str]] = None) -> "Tokenizer":
        """
        一个类方法，从文件加载词汇表和合并规则来构造分词器实例。

        参数:
            cls: Tokenizer 类本身。
            vocab_filepath (str): 词汇表文件的路径。
            merges_filepath (str): 合并规则文件的路径。
            special_tokens (Optional[List[str]]): 特殊词元的字符串列表。

        返回:
            Tokenizer: 一个新的 Tokenizer 实例。
        """
        # TODO: 在这里写入你的实现逻辑
        # 1. 读取 vocab_filepath 文件来构建 vocab 字典。
        # 2. 读取 merges_filepath 文件来构建 merges 列表。
        # 3. 调用 cls(vocab, merges, special_tokens) 来创建并返回实例。
        ...

    def encode(self, text: str) -> List[int]:
        """
        将输入的文本字符串编码成一个 token ID 列表。

        参数:
            text (str): 需要编码的文本。

        返回:
            List[int]: 编码后的 token ID 列表。
        """
        # TODO: 在这里写入你的实现逻辑
        # 核心步骤:
        # 1. 使用正则表达式进行预分词 (pre-tokenize)。
        # 2. 对于每个预分词后的“单词”，将其转换为 UTF-8 字节序列。
        # 3. 在该字节序列上，按顺序应用 self.merges 中的所有合并规则。
        # 4. 将最终的字节块（tokens）转换为 ID。
        # 5. 将所有“单词”的 ID 列表拼接起来。
        ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个字符串的可迭代对象（例如文件句柄）进行编码，并惰性地生成 token ID。
        这对于处理无法一次性载入内存的大文件非常有用。

        参数:
            iterable (Iterable[str]): 一个产生字符串块的可迭代对象。

        返回:
            Iterator[int]: 一个 token ID 的迭代器/生成器。
        """
        # TODO: 在这里写入你的实现逻辑
        # 1. 遍历可迭代对象中的每个字符串块。
        # 2. 对每个块调用 self.encode()。
        # 3. 使用 yield from 将编码后的 ID 逐个产出。
        ...

    def decode(self, ids: List[int]) -> str:
        """
        将一个 token ID 列表解码回文本字符串。

        参数:
            ids (List[int]): 需要解码的 token ID 列表。

        返回:
            str: 解码后的文本字符串。
        """
        # TODO: 在这里写入你的实现逻辑
        # 1. 遍历 ID 列表，使用 self.vocab 查找每个 ID 对应的字节 (bytes)。
        # 2. 将所有字节拼接在一起。
        # 3. 使用 .decode("utf-8", errors="replace") 将完整的字节序列解码为字符串。
        #    errors="replace" 参数可以在遇到无效字节序列时用替换字符来避免程序崩溃。
        ...
