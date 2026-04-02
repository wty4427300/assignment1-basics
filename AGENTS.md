# Repository Guidelines

## Project Structure & Module Organization
`cs336_basics/` contains the assignment implementations, including tokenizer and BPE code in `tokenizer.py` and `bpe_training.py`. `tests/` contains the grading-style test suite, adapters, fixtures, and snapshot data. The assignment handout lives at `cs336_spring2025_assignment1_basics.pdf`. Top-level notes such as `README.md`, `CHANGELOG.md`, and implementation summaries are for developer context, not runtime code.

## Build, Test, and Development Commands
Use `uv` for all local work.

- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_train_bpe.py -q`: run only BPE-related tests.
- `uv run ruff check .`: run linting using the repo Ruff config.
- `uv run python -m cs336_basics.<module>`: run a module directly when needed.

If data-dependent tasks are added later, download corpora into `data/` as described in `README.md`.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and explicit type hints where practical. Keep implementations straightforward and close to the assignment spec. Prefer small helper functions for reusable logic. Ruff is configured with a 120-character line limit; do not introduce formatting that fights that limit.

Keep comments sparse and high-value: explain invariants, tricky byte-level logic, or test-sensitive behavior rather than restating obvious code.

## Testing Guidelines
Tests use `pytest`. Add or update focused tests in `tests/` when changing behavior. Name new tests `test_<feature>.py` and test functions `test_<scenario>()`. For tokenizer or BPE changes, run the targeted tests first, then `uv run pytest` before submitting. Preserve snapshot and fixture files unless the expected behavior intentionally changes.

## Commit & Pull Request Guidelines
Recent history uses short, task-focused commit messages such as `整理代码` and `通过不了speed的bpe`. Follow that pattern: keep commits small, descriptive, and scoped to one change. In pull requests, include a brief summary, the files or modules affected, and the exact test commands you ran. Link the relevant assignment issue or bug if one exists. Screenshots are unnecessary unless the change affects rendered output.

## Agent-Specific Notes
Do not revert unrelated user changes. When editing code, prefer minimal patches that preserve assignment behavior and grading expectations.
