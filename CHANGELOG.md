# Changelog

## 0.8.5 (2026-11-10)
- Speed test in the Tests tab, matched with llama-bench: reading a ~128- and ~512-token prompt from an empty conversation and writing exactly 64 and 512 tokens, twice each after a warm-up, per routing mode (`speed-*.json`).
- `ignore_eos` generation parameter: write exactly `max_tokens` tokens.
- `baseline_llama.bat` also measures llama.cpp on 512-token reads and writes.
- `tools/summarize_results.py` prints the speed tests, every baseline file and every fidelity run, and checks that exact routing matched across runs.
- `docs/results`: the matched speed test, llama.cpp's 512-token runs, and a second fidelity run (exact routing identical to the first on all 10,840 tokens).
- `docs/paper`: the paper (PDF and LaTeX source).

## 0.8.4 (2026-11-09)
- Evaluators' own test suites: import a suite file (kept byte for byte) or write one in the page, with a self-check of example answers; edit, download, delete.
- Every suite is identified by the SHA-256 of its file, shown in the page and recorded in each results file (`suite_sha256`, `suite_author`); results files can be downloaded.
- The answer line may be written in English ("Answer:") as well as Arabic ("الجواب:").
- `tests/template-en.json`: an English example of each check.

## 0.8.3
- The interface is in English by default, with Arabic as a choice (switch at the top right); the README is in English, with `README.ar.md` in Arabic.

## 0.8.2
- Fidelity test: answers written with exact routing are read back token by token under each routing mode; per token, agreement with the exact model's top choice, KL divergence, clear disagreements and perplexity change (`POST /api/tests/fidelity`).
- Rafid keeps the computer awake while it works.
- `baseline_llama.bat`: plain llama.cpp (`llama-bench`, mmap, experts on the CPU side) on the same model, for comparison.

## 0.8.1
- With thinking off, the `<think>` and `</think>` tokens are barred from sampling (the model reopened thinking on its own and spent the whole answer in it).
- The test runner never feeds an empty answer back into the conversation; it continues from the example answer and saves the raw text.

## 0.8.0
- Tests: multi-turn conversation suites graded automatically, run once per routing mode, saved after every answer and resumable. Bundled Arabic suite of 9 conversations and 84 questions.

## 0.7.x
- Reading the question while the user types (`POST /api/prefill`), with state snapshots for the recurrent layers of Qwen3.5.
- "Light" cache-aware routing became the default.

## 0.6.x
- "Recent frequency" (decayed LFU) cache policy as the default; prefetch from the next layer's gate; graded cache-aware routing (light, medium, strong).
- The expert cache never exceeds the RAM that is really free; the answer line names the routing that produced it; mathematics in Arabic text is laid out correctly.

## 0.1 – 0.5
- The planner, hardware probe and GGUF reader; the expert fingerprint and its analysis; experts streamed from the disk with a fixed RAM cache, unbuffered parallel reads and a slim model copy for Windows.
