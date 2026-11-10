# Rafid: a home AI platform

**English** · [العربية](README.ar.md)

Rafid is a C++ program that runs AI models on your own computer. It splits each model between the **GPU** and **RAM**, and for mixture-of-experts (MoE) models it can keep the experts on the **disk** with a cache of them in RAM. That lets it run models far larger than the computer's memory: Qwen3.5-397B-A17B (115 GB) runs on a laptop with 31 GB of RAM and an 8 GB GPU at 4.8–6.0 tokens/s.

It has three parts:

1. **The planner** reads the model file and measures your computer's real memory speeds (RAM, GPU memory, the link between them, the disk). It then decides where each part of the model lives and estimates the speed. If loading fails for lack of memory, it retries automatically with a lighter plan.
2. **The expert fingerprint.** For MoE models, Rafid records which experts the router picks in each layer for each token. It then analyses this record to answer three questions: is the work concentrated in a few experts? How well would a cache of the hot experts work? Can the next layer's experts be predicted?
3. **The interface** is a page that opens in the browser, in English or Arabic (switch at the top right). It holds the chat, the machine and plan screen, the fingerprint screen and the tests.

The compute engine (CPU kernels, CUDA and GGUF reading) is [llama.cpp](https://github.com/ggml-org/llama.cpp), build `b12167` (MIT licence; see `THIRD_PARTY_NOTICES.md`).

| | |
|---|---|
| **Run it on Windows** | download the release `rafid-0.8.5-windows-x64-cuda13.zip`, unzip, double-click `run_rafid.bat` ([below](#running-on-windows-no-build-needed)) |
| **Build it** | `build_windows.bat` (Visual Studio + CUDA), `./build_linux.sh`, or `tools/build_windows_mingw.sh` from Linux ([below](#rebuilding-rafidexe-development-only)) |
| **The paper** | `docs/paper/rafid-paper.pdf` (LaTeX source alongside) |
| **The paper's data** | `docs/results/`, with `docs/REPRODUCE.md`; `python tools/summarize_results.py docs/results` prints every table |
| **Licence** | MIT, © 2026 Zaid A. Lateef A. Kareem (`LICENSE`) |
| **Citing** | `CITATION.cff` |
| **Changes** | `CHANGELOG.md` |

---

## Running on Windows (no build needed)

The Windows release is ready to run: its `bin` folder holds `rafid.exe` and the official llama.cpp libraries (build b12167, CUDA 13.4 for NVIDIA cards). The source repository has no `bin` folder; build it, or take the release. **You need neither Visual Studio nor the CUDA Toolkit.**

All you need:
- **A recent NVIDIA driver** (version 580 or newer), from the NVIDIA App or nvidia.com.
- **The Microsoft Visual C++ Runtime.** Most computers have it. If it is missing, `run_rafid.bat` says so and gives Microsoft's official download link.

### Steps

1. Double-click `run_rafid.bat`. A black window with the run log appears: keep it open, since closing it stops Rafid. Rafid measures memory speed for a few seconds, then the interface opens in the browser at `http://127.0.0.1:8080`.
2. In **Machine & model**, check that "Compute backends" lists **CUDA** and that your GPU is shown.
3. Press "Show LM Studio models" to use the models you already downloaded, or put `.gguf` files in the `models` folder.
4. Choose a model. The **plan** appears at once: a map of the layers (each brick is a layer, its top half attention and its bottom half experts; gold means GPU and teal means RAM), each memory's use, and the expected speed.
5. Press "Load the model", then start chatting.
6. After loading, try "Pick the fastest thread count" once. Your CPU has performance and efficiency cores, and sometimes the performance cores alone are faster.

Tips: keep the laptop on its charger and in high-performance mode. For a manual split, choose "Manual" and move the slider to see its effect on speed before loading.

### Chatting with thinking models (Qwen3.5, Qwen3.6 and similar)

- **"Think before answering"** appears next to "New chat" when the model's template can switch thinking off (`enable_thinking`). With it off, the model answers directly, which saves hundreds of tokens per answer and matters a lot with large, slow models. Since 0.8.1, the `<think>` and `</think>` tokens are barred while thinking is off. In the large test, Qwen3.5 reopened a thinking block on its own after the empty one, on a simple question, spent all 700 tokens in it and wrote no answer.
- **The conversation is not re-read every turn.** Qwen3.5 has recurrent layers (Gated DeltaNet) whose state cannot be wound back. So Rafid keeps snapshots of that state (about 150–200 MB for large models, two at most): at the end of each answer, at the start of the question you are typing, and before the part it read speculatively (next point). The next turn continues from a snapshot instead of reading the whole conversation again. When there is no room, the speculative snapshot goes first, then the end-of-answer one, and the start-of-question one last. The line under an answer says "reused N tokens of the conversation". To change the number of snapshots: `RAFID_CHECKPOINTS` (0 turns them off).
- **Rafid reads your question while you type it (since 0.7).** Most of the wait before the first word is reading experts from the disk (on the 397B a 50-token question reads about 25 GB and takes 5–7 s), and the disk is idle while you type. So the page sends what you type as you go, and Rafid reads all complete words except the last two tokens (which may change with what follows). If you stop typing for a second and a half, it reads the rest and the end of the template as if you had sent it, up to the probabilities the answer starts from. If you then send the question as it is, the answer starts at once; if you type more, it goes back to the snapshot before the speculative part; if you delete words it had read, it goes back to the start-of-question snapshot. Sending never waits for this reading: it stops at its next step and the question continues from where it got to. Under the answer you see "read N tokens while you typed", then "reading the rest" or "answer started at once". It can be switched off in "Generation settings". (`POST /api/prefill`. The result matches reading the question in one go except for rounding; the test compares the probabilities.)
- If the model wrote text with tokens other than the ones the tokenizer would normally choose, Rafid builds the next prompt on the same stored tokens as long as the text is the same, so nothing in memory is lost.

### Rebuilding rafid.exe (development only)

- **From Linux with MinGW** (how the current build was made): `tools/build_windows_mingw.sh <llama.cpp b12167 source> <DLL folder> <output folder>`
- **On Windows** with Visual Studio 2022 and CUDA Toolkit 12.8+: `build_windows.bat` (builds llama.cpp from source).
- **On Linux:** `./build_linux.sh`

## Suggested models

Dense models (such as Qwen3-8B in LM Studio) are fine for chatting and for checking that CUDA works. The **expert fingerprint** needs a MoE model. Look for a GGUF version on Hugging Face; `Q4_K_M` is enough. These are well-known examples, and newer MoE models may exist:

| Model | Size, roughly | Why |
|---|---|---|
| Qwen3-30B-A3B | 18–19 GB | A strong MoE model that suits an 8 GB GPU + 32 GB RAM. **Best for trying the fingerprint** |
| gpt-oss-20b | 12–13 GB | A very fast MoE on this kind of machine |
| Small Granite MoE (such as 3B-A800M) | about 2 GB | To check quickly that the build works before downloading a large model |

For models split into parts (`-00001-of-00005.gguf`), keep all parts in the same folder; Rafid shows them as one model.

## Trying the fingerprint (the first step of our plan)

The goal is to check the assumption behind running 400B models: is your use of experts concentrated enough for a cache to work?

1. Load Qwen3-30B-A3B. The fingerprint is recorded automatically; you can stop it in "Expert fingerprint".
2. Use it for a week as usual: Arabic, programming, everyday questions. You need at least 2,000 generated tokens, ideally 20,000 or more.
3. In **Expert fingerprint**, press "Analyse the fingerprint" and read the three answers:

| Question | Encouraging result | What it means |
|---|---|---|
| Concentration | under 35% of experts cover 80% of the calls | a fixed cache (the fingerprint) will work well |
| Cache | 70% or more hits with a cache of 20% of the experts | experts from the disk with a RAM cache can run at an acceptable speed |
| Prediction | clearly better accuracy than "guessing by popularity" | prefetching the next layer will hide the disk read time |

4. Use "What if you ran a 400B model?" to see what your results mean for a large model on your computer (measure the disk speed first in the machine section).

Every cache method is trained on the first half of your fingerprint and tested on the second half, so results do not look better than they are.

**Privacy:** the fingerprint stores expert numbers only (which expert was chosen in which layer), never the text of your chats. It is kept in the `profiles` folder:
- `<model>.rfprof`: the running counters (the long-term fingerprint).
- `<model>.rftrace`: the sequence of choices for generated tokens, used to simulate caches, at most 1 GB.

## Phase two: experts from the disk

In this mode Rafid keeps attention and the shared parts on the GPU, leaves **all experts on the disk**, and keeps a cache of them in RAM **of the size you choose**. This is how a model larger than your computer's memory runs.

How it works:
- **The cache is a fixed set of slots per layer**, reserved once at load and kept in RAM (locked there if Windows allows). A new expert is written over an old expert's slot, so Rafid never asks Windows for new memory, or returns it, while generating.
- Before each layer Rafid knows which experts the router chose, and fetches the missing ones from the disk **with unbuffered reads that bypass the Windows file cache**, split into 256 KB pieces read by 16 threads in parallel.
- Rafid then replaces the expert numbers the router chose with **their slot numbers**, between computing the expert weights and computing the experts themselves, so the computation reads the slots directly with no copying.
- **Reading the question:** a prompt batch needs many experts of each layer, so they are gathered in a fixed buffer the size of a whole layer (from the cache when there, otherwise straight from the disk). While a layer computes, **Rafid uses the next layer's gate to guess its experts for all the question's words** (a sample of up to 128 words) and gathers them in a second buffer, so the next layer has little left to wait for. The words of one question share most of their experts: a 113-token question on the 397B needed about 44% of each layer's experts, so guessing costs less than gathering the whole layer (the old way, kept only when gate guessing is off). Under the answer you see "X% of the question's experts gathered ahead". Batches go up to 2,048 tokens.
- **"Recent frequency" policy (the default since 0.6):** each expert has a use counter that halves every 32 tokens, and the expert with the lowest counter is evicted. Simulated on the fingerprints of Qwen3.6-35B, Qwen3.5-122B and Qwen3.5-397B, it hit more than LRU at every size (+1.5 to +4.4 points), which means 7–12% less reading from the disk. A correct guess counts as one use, and a wrong guess is evicted first.
- **LRU policy:** each layer keeps the most recently used experts. Your fingerprint fills the slots with the most used experts at the start (in every policy).
- **"Pinned fingerprint + most recent" policy:** the experts used most in your fingerprint stay fixed (70% of each layer's slots).
- **Guessing the next layer's experts** while the current layer computes, in two ways:
  - **From the gate (the default):** Rafid takes the next layer's own router and applies it to the model's current state, on a side thread. Its accuracy on Qwen3.6 was 81%.
  - **From the fingerprint:** relies on which experts followed each other in your history. Its accuracy was 39%.
  A guess never evicts experts used by the current word, and a wrong guess is cancelled if its read has not started, or sent to the end of the cache.
- **Cache-aware routing (approximate; "light" is the default since 0.7):** if one of a token's lowest-weight experts is not in RAM, Rafid uses instead the expert in RAM that the router weighs most, provided its weight is close to the missing expert's. This happens before the expert weights are gathered, so the replacement gets its real weight from the router. The highest-weight expert is never replaced. Three levels:
  - **Light:** the lowest 30% of the experts (for 10 experts: ranks 8–10), and the replacement needs at least 70% of the missing expert's weight.
  - **Medium:** the lowest 50%, and 50% of the weight.
  - **Strong:** the lowest 70%, and 30% of the weight.
  This **changes the computation slightly from the full model**. The real measurement shows its effect in the "Agreement" column, and the fidelity test (below) measures it token by token. "Off" gives exactly the model. The line under each answer names the routing that made it, for example "exact model" or "medium routing: swapped 51 experts per token", so a screenshot alone tells which mode gave the answer.

### How to try it

1. Load Qwen3.6 in the normal mode first and chat with it a little, so it has a fingerprint.
2. In "Machine & model", in the model options, turn on **"Leave the experts on the disk, and keep a cache of them in RAM"**. Choose the cache size (2 or 3 GB, say), the policy and the guessing method, then "Load the model". The plan's speed estimate uses the hit rate expected from your fingerprint.
3. Chat as usual. Under each answer you see the **cache hits**, **how many MB were read from the disk per token**, **the share of time the model waited for the disk**, the **guess accuracy**, and how much the question read from the disk and how fast.
4. In "Expert fingerprint" → **"The real measurement"**: choose cache sizes, policies, guessing methods and routing, then "Start the measurement". A table shows **expected against real speed** for each trial, the hit rate, guess accuracy and the real disk speed. Hover over the gate's guess accuracy to see the accuracy two layers ahead.
   - **Agreement:** the first exact trial writes the text, and every later trial reads **exactly the same text** (teacher forcing). At each position Rafid compares the trial's most likely token with the full model's. Exact trials give 100% (a check that the cache changes nothing), and cache-aware routing gives less. Hover over the number to see how many experts were swapped per token.
   - The expected speed includes the effect of gate guessing and is calibrated on the 122B and 397B measurements: the CPU reaches about 40% of RAM speed on compressed experts, and reading about 80% of the disk's sequential speed.
5. "Current session settings" changes the cache size, policy, guessing and routing for the chat without reloading.

### Qwen3.5-397B-A17B on a laptop (UD-IQ2_XXS, 115 GB, 31 GB RAM)

| Cache | Guessing | Speed | Hits | Read/token | Disk wait |
|---|---|---|---|---|---|
| 10% (10 GB) | none | 3.7 tok/s | 61% | 814 MB | 61% |
| 10% (10 GB) | from the gate | 4.7 tok/s | 72% | 986 MB | 51% |
| 14% (14.3 GB) | none | 4.1 tok/s | 67% | 687 MB | 58% |
| 14% (14.3 GB) | from the gate | 5.2 tok/s | 76% | 844 MB | 48% |

With a 14% cache and gate guessing, a token takes 192 ms: about 100 ms of CPU compute and 92 ms waiting for the disk (reading 844 MB per token at 5.3 GB/s).

**Rafid 0.6 (gate guessing in every row, 48 tokens per trial):**

| Cache | Policy | Routing | Speed | Agreement | Read/token | Disk wait |
|---|---|---|---|---|---|---|
| 10% (10 GB) | LRU | exact | 4.7 tok/s | 100% | 985 MB | 50% |
| 10% (10 GB) | recent frequency | exact | 4.8 tok/s | 100% | 940 MB | 48% |
| 10% (10 GB) | recent frequency | light | 5.5 tok/s | 96% | 798 MB | 42% |
| 20% (20 GB) | LRU | exact | 5.9 tok/s | 100% | 670 MB | 41% |
| 20% (20 GB) | recent frequency | exact | 6.1 tok/s | 100% | 635 MB | 39% |
| 20% (20 GB) | recent frequency | light | 6.9 tok/s | 98% | 527 MB | 33% |

- Recent frequency reads 5–7% less than LRU and is 0–8% faster.
- Light routing is 13–26% faster and matches the full model at 96–98% of positions.
- Every exact trial matched 100%: the cache and reading fixed text change nothing.
- **Lesson:** a 30% cache (30 GB) on a laptop with 31 GB of RAM dropped to 1–1.6 tok/s, because Windows started moving memory to the disk. Windows' "available RAM" includes file cache and pages already moved out, so it can promise more than the machine has. Since 0.6.1 the cache never exceeds "total RAM − the system's share (an eighth of RAM, at least 3 GB) − the rest of the model", and Rafid warns when it cannot lock the cache in RAM.

**Question test (Rafid 0.6.6, 20% cache, temperature 0, a new chat for each question):**

| Question | Full | Light | Medium | Strong |
|---|---|---|---|---|
| A bat and a ball for 1,100; the bat costs 1,000 more | ✓ 5.2 | ✓ 6.1 (exactly the same text) | ✓ 6.8 | ✓ 6.8 |
| Sunday + 100 days | ✓ 5.9 | ✓ 6.5 | ✗ 6.9 (remainder 6 instead of 2) | ✓ 7.3 |
| Three boxes, all labelled wrong | ✓ 5.3 | ✓ 6.1 | ✓ 6.9 | ✓ 7.1 |

(Speeds in tok/s.) Light answered 8 questions, all correctly, 10–17% faster than the full model. Medium was wrong once and strong not once in three; errors at these levels are chance: the more experts are swapped, the more likely an answer drifts from the full model. One correct answer does not prove a level safe.

### What the first two measurements taught us (Qwen3.6-35B-A3B, 10–30% cache)

- **The hit rate expected from your fingerprint matched the real one almost exactly.** The cache simulation can be trusted.
- **LRU beat the pinned fingerprint** at every size, so it became the default (later replaced by recent frequency). A separate cache per layer gives nearly the same hit rate as one shared cache (0–1.6 points apart on your fingerprint).
- **Gate guessing is 81% accurate**, against 39% from the fingerprint.
- **The bottleneck was not the disk but Windows memory management.** When every new expert took new memory pages and returned them on eviction, the wait grew by a fixed amount per expert (about 1.25 ms), and the real speed stalled at 1.0–1.6 GB/s however many threads were used. That is why the cache became fixed slots that are neither requested nor returned while generating.

### How to read the results

- If the real speed is close to the expected one, the simulation can be trusted, and the "What if 400B" numbers can guide decisions.
- If the real speed is much lower, the gap points at what the simulation did not count, such as small scattered reads or waiting for the CPU.
- Comparing "none" with the two guessing methods shows whether guessing really saves time or spends disk speed on wrong reads.

Notes:
- In this mode **the CPU computes all experts**, so it is slower than the normal mode even with a full cache. Its purpose is measurement, not everyday chat.
- Besides the cache, Rafid reserves two buffers for reading the question, each the size of a whole expert layer (about 650 MB for Qwen3.6); they appear in the memory plan.
- **A slim copy of the model (Windows):** llama.cpp opens the model file through memory mapping (mmap), and Windows (NTFS) reads a file opened that way **one request after another**, however many threads ask. So in this mode Rafid writes, once, a copy of the model without the experts in the `slim` folder (the same apparent size, but the experts are holes that take no space: about 2.5 GB on disk for Qwen3.6) and gives it to llama.cpp, while it reads the experts from the original file, which nobody maps. You can delete the `slim` folder at any time; it is rebuilt when needed. To turn this off: `RAFID_NO_SLIM=1`.
- **Cache size:** the default is 20% of the experts, but never more than fits in the RAM left after the rest of the model (the planner computes it and shows it next to the slider: "the largest that fits in RAM now"). Your choice is kept per model, and if you choose a size that does not fit Rafid shrinks it and warns you, because a cache Windows has to move to the disk is slower than none. "The real measurement" likewise offers only sizes that fit, with a "largest that fits" option.
- "Measure disk speed" also measures scattered 256 KB reads with one and 16 threads, then with 16 threads and the file memory-mapped, which shows whether parallel reading works on your computer. The result is saved in `profiles/hardware.json`, so the expected speed stays right after restarting Rafid (it is ignored if the CPU changes, and measured again).
- A copy of the layers' routers is kept in RAM for guessing (about 40 MB for Qwen3.6).
- Measurement trials do not enter your fingerprint.
- Safety net: if any computation reaches an expert that did not go through the main path, Rafid loads it at once and redoes that step, so results stay correct. The count of such cases appears in the log and should be zero.
- **Threads:** in this mode the CPU computes all experts, so Rafid uses every core by default (performance and efficiency). On a Core Ultra 9 275HX with Qwen3.5-122B: 8 threads 13.1 tok/s, 16 threads 15.8, 24 threads 18.9 (with every expert a cache hit). In the other modes the default stays the performance cores only.
- For experiments: `RAFID_STREAM_IO_THREADS` (reader threads), `RAFID_STREAM_CHUNK_KB` (piece size) and `RAFID_STREAM_BUFFERED=1` (reads through the Windows file cache).

## Tests: continued conversations, checked automatically (since 0.8)

The "Tests" section runs a set of conversations on the loaded model and grades them itself. Each conversation is a series of questions, and each question goes to the model with the whole conversation, including its own earlier answers, as in real use. If it gets one question wrong, it builds on that mistake in the following ones.

- **The bundled suite** `profiles/tests/suite-ar.json`: nine conversations and 84 questions in Iraqi Arabic, Modern Standard Arabic and English. It has a shop ledger that accumulates with corrections in the text, a family whose details change, JSON and tool calls, a long (fictional) hospital report with buried numbers, a staff rota whose rules change until it becomes impossible, standing instructions that change (a signature at the end of every answer), code, a car trip with an average-speed trap, and the history of Baghdad.
- **Checks:** each question has checks: the number on the "الجواب:" (answer) line (accepting ١٬٧٠٠٬٠٠٠, 2.9 million and negatives), words that must or must not appear (Arabic is compared without diacritics, with أ/ا, ة/ه and گ/ك treated alike), valid JSON and its values, counts of bullets, words and commas, the language, an order, and the signature at the end. Every question in the suite has an example of a correct answer, and a self-test checks that every example passes and that broken answers fail.
- **Running:** temperature 0 and no repetition penalty (the penalty breaks numbers with repeated zeros), with a 700-token limit per answer. The whole test runs once for each routing mode you choose (off/light/medium/strong), then the routing goes back to what it was. Chat is paused during the test.
- **Results** are saved after every answer in `profiles/tests/results/`: the question, the answer, each check's result and the speed. If the test is interrupted, "Continue" rebuilds the conversation from the saved answers and goes on from the same question. The table shows ✓ or ✗ for each question in each mode, and how many answers were word-for-word the same as the first mode's.
- **An answer that never got written (since 0.8.1):** if an answer is cut at the limit while still thinking, or comes back empty, it is not put back into the conversation empty: an empty answer in the conversation teaches the model to answer nothing (in the first run every later question came back empty). The conversation continues from the example answer, the question counts as failed, and the whole text the model wrote is saved so we can see what happened ("The whole text it wrote" under the answer). The table cell gives the reason: cut, opened a thinking block, wrote no answer. If you choose "Think before every answer", 3,000 tokens are added to the limit for thinking.
- **A new suite:** a JSON file of the same shape in the `profiles/tests` folder (`tools/make_test_suite.py` builds the bundled suite and checks its arithmetic).

### Your own tests (for evaluators)

Results are more credible when the questions come from someone else. In "Tests", an evaluator can bring their own questions in two ways, without touching the code:

- **"Import a suite file":** a `.json` suite written elsewhere (shape below, or start from `tests/template-en.json`). It is saved **byte for byte**.
- **"Write a new suite":** an editor in the page. Each conversation is a list of questions; each question has its checks and, optionally, an example of a correct answer. "Check the example answers" runs the grader on the examples, so the evaluator sees that the checks pass a right answer before running anything. "Ask for a final “Answer:” line" adds the request that the answer-line checks need. Saved suites can be edited, downloaded or deleted later; the bundled suite cannot be changed.

**How the evaluator can verify the run.** Every suite is identified by the **SHA-256 of its file**, shown under the suite and recorded in every results file (`suite_sha256`, with `suite_author`). The evaluator computes the hash of the file they sent (`certutil -hashfile suite.json SHA256` on Windows, `sha256sum suite.json` elsewhere) and compares. "Download the results file" gives the whole run: every question, the model's full answer, every check's result and the speed, so the evaluator can re-grade it independently.

**A blind run:** the evaluator keeps the questions until the run, sends the file, the model's owner imports and runs it, and returns the results file. The hash in the results proves the questions were not changed.

**Suite format.** A JSON object with `title`, `author`, optional `about`, and `conversations`; each conversation has optional `id`, `title` and `system` (instructions), and `turns`; each turn has `user` (the question), `checks`, and optional `ex` (an example answer) and `max_tokens`. A question passes when all its checks pass. Checks:

| Check | Passes when |
|---|---|
| `{"num": 425}` | the number is on the reply's last "Answer:" (or «الجواب:») line; `"tol": 0.5` allows a margin; `"where": "any"` looks anywhere. Numbers like 1,650,000, ١٬٦٥٠٬٠٠٠ and "2.9 million" are understood |
| `{"nums": [3, 4]}` | all these numbers appear anywhere |
| `{"has": ["Tokyo"]}` | all these texts appear (case, Arabic diacritics and letter variants ignored); `"where": "line"` limits it to the answer line |
| `{"has_any": [...]}`, `{"not_has": [...]}` | at least one appears / none appears |
| `{"order": ["first", "second"]}` | these appear in this order on the answer line (`"where": "any"`: anywhere) |
| `{"words": 5}`, `{"max_sentences": 1}`, `{"bullets": 3}`, `{"no_commas": true}` | exact word count, at most N sentences, exactly N bullet points, no commas |
| `{"lang": "en"}` | at least 80% of the letters are English (`"ar"`: Arabic; `"min"` changes the share) |
| `{"json": {"strings": ["red"]}}` | the reply holds valid JSON (a fenced block or the first object or array) containing these texts; more JSON checks are in the bundled suite |
| `{"ends_with": "— Rafid"}` | the last line is this (a signature) |
| `{"any": [check, check]}`, `{"all": [...]}` | one of / all of several checks |

### Comparison with plain llama.cpp

`baseline_llama.bat` measures plain llama.cpp's speed on the same model and the same computer. It uses the same build Rafid uses (b12169) and its own benchmark tool `llama-bench`, set up the usual way for a model larger than memory: the layers on the GPU (`-ngl 99`, falling back to 40 then 0 if they do not fit), and the experts read from the disk through mmap (`-ncmoe 99`). It measures writing 64 tokens and reading a 128-token question (twice each after a warm-up), then writing and reading 512 tokens once each. The speed test in the Tests tab measures Rafid the same way. Results go to `profiles/tests/results/baseline-llamacpp-*.json`, with a `.log`. Close Rafid first. Programs that guard memory, such as LM Studio with its default settings, refuse to load the model at all.

### The fidelity test: how far routing moves the model (since 0.8.2)

Cache-aware routing acts only while an answer is written (one token at a time), not while a question is read (in batches). So a multiple-choice test with a one-letter answer measures nothing: the letter comes from reading the question, and is the same in every mode. And counting correct answers on 84 questions cannot show a small difference.

So the fidelity test takes the answers written with routing off in the conversation test (for each conversation, the latest complete run without thinking) and has each mode read them token by token while routing, inside the same conversation. "Off" is read first, as the reference, and keeps for each token its choice and its 32 most likely tokens. Then, in each mode, for each token:
- **Same choice:** whether the most likely token is the same as "off"'s choice.
- **Distance:** KL(off ‖ mode) in nats, over the reference's 32 most likely tokens and the rest as one group (a lower bound on the full KL, and close to it).
- **Tokens clearly apart:** the share of tokens whose distance is over 0.1.
- **Text difficulty:** the increase in perplexity on the reference answer.
- And the speed, disk reads and cache hits, for answer tokens only. Questions are read in batches, exactly, in every mode.

Every mode starts from the same cache (emptied and refilled from the fingerprint). Routing depends on which experts have arrived from the disk at that moment, so repeating the test gives close but not identical numbers; "off" repeats exactly. Results go to `profiles/tests/results/fidelity-*.json`, with the totals of every question in every mode. (`POST /api/tests/fidelity`; needs a model loaded with streaming.)

Since 0.8.2 Rafid also keeps the computer from sleeping while it works (an answer, a test or a measurement). On one test night, a single answer waited five hours because the computer slept. Closing the lid still puts it to sleep.

## Commands

```
rafid                       the server + the interface
rafid hw                    a hardware report with memory speed measurements
rafid plan  <model.gguf>    the plan and expected speed, without loading
rafid analyze <model.gguf>  analysis of the saved fingerprint (JSON)

--models-dir DIR   --profiles-dir DIR   --model PATH   --ctx N
--cpu-moe N        the experts of the first N MoE layers in RAM
--ngl N            number of layers on the GPU (manual)
--moe-cache        experts in RAM + a cache of hot experts on the GPU (experimental in llama.cpp)
--threads N        --port P   --no-browser   --no-bench   --verbose
```

## Common problems

- **The GPU does not appear, or CUDA is missing from "Compute backends":** update the NVIDIA driver to 580 or newer, then restart Rafid. Check that these files are in `bin`: `ggml-cuda.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` and `cudart64_13.dll`.
- **A message about msvcp140.dll:** install the Microsoft Visual C++ Runtime from the link `run_rafid.bat` shows.
- **Loading fails for lack of memory:** Rafid retries automatically up to 6 times with a lighter plan. If it keeps failing, reduce the context size. The details are in the "Run log".
- **Port 8080 is busy:** `run_rafid.bat --port 8090`
- **Speed is lower than estimated:** the estimate is built on measured memory speeds and approximate efficiency factors. Check high-performance mode and the charger, and try tuning the threads.
- **The interface language:** the switch at the top right of the page chooses English or العربية; the choice is remembered in the browser.

## Project layout

```
src/hw_probe     hardware detection and speed tests for RAM, GPU memory, uploads and the disk
src/gguf_info    reading the GGUF file and classifying each tensor (attention, router, shared experts, experts...)
src/planner      the plan: where each part lives, the speed estimate, automatic fallback when memory runs short
src/profiler     recording the router's choices from the compute graph (ffn_moe_topk)
src/analyzer     concentration, cache simulation (fixed, LRU, SLRU, recent frequency), cross-layer prediction
src/expert_store experts from the disk: reserved memory, unbuffered reads, cache, prefetch, safety net
src/engine       loading the model by the plan, chat with conversation-memory reuse, the fidelity test
src/server       HTTP API with token streaming (SSE); the page is built into the program
web/index.html   the interface, English and Arabic (with the test runner and grader)
tests/           test suites (copied to profiles/tests)
bin/             rafid.exe and the llama.cpp and CUDA libraries, ready for Windows
tools/           build rafid.exe for Windows from Linux (MinGW), build the bundled suite, summarize results
docs/            the paper (docs/paper), REPRODUCE.md and its raw results (docs/results)
```

## Next steps

Phase two started with the "experts from the disk" mode above. After the real measurements, we choose among:
1. **Ordering the experts on the disk by use**, so their reads are adjacent and larger.
2. **Mixed compression:** hot experts at 4-bit and rare ones at 2-bit, so less data is read from the disk.
3. **Speculative decoding** with a small model on the GPU that guesses tokens and experts together.
4. A real large MoE model, once the numbers are confirmed.

## Licence and citation

Rafid is released under the MIT licence (`LICENSE`), © 2026 Zaid A. Lateef A. Kareem. The third-party software it builds on and ships is listed in `THIRD_PARTY_NOTICES.md`. If you use Rafid or its results, please cite it (`CITATION.cff`).
