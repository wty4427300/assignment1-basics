# Gemini 上下文: CS336 作业 1 - BPE 分词器

## 项目概述

本项目是 CS336 课程的一项 Python 作业，专注于自然语言处理的基础知识。核心任务是从头开始实现一个字节对编码（Byte-Pair Encoding, BPE）分词器。

该项目包含两个主要部分：
1.  **BPE 训练器 (`bpe_training.py`):** 一个用于训练 BPE 模型的脚本。它接收一个原始文本语料库，基于 BPE 算法学习一个词汇表和一套合并规则，并将它们保存到文件中。
2.  **分词器 (`tokenizer.py`):** 一个 `Tokenizer` 类，可以从训练好的词汇表和合并规则文件进行实例化。它提供了将文本 `encode` (编码) 为词元 ID 和将词元 ID `decode` (解码) 回文本的方法。

该实现是字节级（byte-level）的，意味着它操作的是 UTF-8 字节而不是抽象的字符。这使得分词器非常健壮，能够处理任何文本而不会产生“未知词元”的错误。项目使用了 `regex` 库，因其强大的、支持 Unicode 的模式匹配能力，这对于基于 GPT-2 分割模式的预分词步骤至关重要。

## 构建与运行

该项目使用 `uv` 进行 Python 环境和依赖管理。所有依赖项都在 `pyproject.toml` 文件中列出。

### 环境设置

1.  **安装 `uv`:**
    ```sh
    pip install uv
    ```
2.  **安装依赖:** `uv` 在运行命令时会自动处理依赖安装。不需要显式执行 `uv pip install -r requirements.txt` 这样的步骤。

### 运行测试

验证实现的主要方式是使用 `pytest` 运行提供的测试套件。测试会将自定义分词器的输出与参考的 `tiktoken` (GPT-2) 分词器进行比较。

运行测试：
```sh
uv run pytest
```

测试文件位于 `tests/` 目录下。关键的测试文件包括：
-   `tests/test_train_bpe.py`: 测试 `train_bpe` 函数。
-   `tests/test_tokenizer.py`: 测试 `Tokenizer` 类，包括编码/解码的往返一致性以及与 `tiktoken` 的对比。

### 训练新的分词器

在一个文本文件（例如 `corpus.en`）上训练一个新的 BPE 分词器：
```sh
# 训练逻辑位于 cs336_basics/bpe_training.py
# 通常会从另一个脚本或模块中调用它。
# 使用示例 (假设):
uv run python -c "from cs336_basics.bpe_training import train_bpe; vocab, merges = train_bpe('tests/fixtures/corpus.en', vocab_size=1000, special_tokens=['<|endoftext|>']); print(merges)"
```

## 开发规范

*   **环境管理:** 严格使用 `uv`。
*   **核心逻辑:** 主要的实现文件是 `cs336_basics/tokenizer.py` 和 `cs336_basics/bpe_training.py`。`tokenizer.py` 中的 `Tokenizer` 类是不完整的，需要补充实现。
*   **测试:** 项目是测试驱动的。所有功能必须通过 `tests/` 目录下的测试。测试依赖于 `tests/adapters.py` 中的适配器函数，以连接测试套件和学生的实现。
*   **预分词:** 一个关键步骤是使用 `cs336_basics/bpe_training.py` 中提供的特定 GPT-2 正则表达式模式进行预分词。训练脚本使用 `ProcessPoolExecutor` 来并行化此步骤以提高效率。
*   **特殊词元:** 像 `<|endoftext|>` 这样的特殊词元必须被正确处理。它们不受 BPE 合并的影响，并被视为原子单元。
*   **代码风格:** 代码风格由 `ruff` 强制执行，其配置位于 `pyproject.toml` 中。行长限制为 120 个字符。