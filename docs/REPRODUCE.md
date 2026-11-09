# Reproducing the paper's numbers

All results in the paper come from one laptop and one model, and the raw files are in `docs/results/`.
`python tools/summarize_results.py docs/results` prints every table from them.

## Setup

| Item | Value |
|---|---|
| Laptop | Intel Core Ultra 9 275HX (24 threads), 31 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU 8 GB, NVMe 6.2 GB/s (`hardware.json`) |
| Model | Qwen3.5-397B-A17B, unsloth `UD-IQ2_XXS` GGUF (4 files, 115 GB) |
| Rafid | 0.8.0 for the first conversation run, 0.8.1 for the rerun of the hospital-report conversation, 0.8.2 for the fidelity test |
| Load options | experts from the disk, expert cache 21.5 GB, "recent frequency" policy, prefetch from the gate, context 65,536 |
| Decoding | temperature 0, repetition penalty 1.0, thinking off, 700 tokens per answer (some turns 900–1,200) |

## The tables

| Paper | How it was made | File(s) |
|---|---|---|
| Accuracy on 84 conversational questions, off and light | Tests → suite `suite-ar.json`, modes off + light, all conversations | `suite-ar-20261108-191139.json` |
| The hospital-report conversation after the thinking-tag fix | same, only "Hospital report" (قراءة تقرير طويل) | `suite-ar-20261108-213349.json` |
| Accuracy, medium and strong | same suite, modes medium + strong | `suite-ar-20261108-215443.json` |
| Fidelity per routing mode | Tests → Fidelity test, modes light, medium, strong (off is the reference) | `fidelity-20261109-081637.json` |
| llama.cpp baseline | `baseline_llama.bat` (Rafid closed) | `baseline-llamacpp-gen.json`, `baseline-llamacpp-pp.json`, `baseline-llamacpp.log` |

Each conversation-test file holds, for every question and mode, the model's full answer, each check's
result, the speed and the expert-cache statistics. The fidelity file holds per-turn sums for every mode.

## Notes

- Routed modes depend on which experts have arrived from the disk at each moment, so a rerun gives close but
  not identical numbers. Exact routing ("off") repeats exactly.
- In the first full run, the hospital-report conversation broke at question 5 in both modes: the model
  reopened a thinking block with thinking off and wrote no answer, and the empty answer, fed back into the
  conversation, made every later answer empty. Version 0.8.1 fixed both causes, and the conversation was
  rerun (`suite-ar-20261108-213349.json`); the summary script uses the newest run of each conversation.
- One question in the medium/strong run took 5.3 hours of wall time because the computer slept; its
  generation took 34 s. Speeds are measured by the engine, not by wall time. Since 0.8.2 Rafid keeps the
  computer awake while it works.
- The llama.cpp numbers are short runs (64 tokens, twice). A longer run and a matched prompt-reading
  measurement are listed as open work in the paper.
