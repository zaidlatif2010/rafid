#!/usr/bin/env python3
"""Prints the tables of the Rafid paper from the results files.

    python tools/summarize_results.py docs/results          (the paper's own data)
    python tools/summarize_results.py profiles/tests/results (your runs)

Conversation tests: for each routing mode and conversation, the newest run of that pair counts (so a rerun
of one conversation replaces it). Fidelity: every finished fidelity file, oldest first, and whether exact routing matched across them. Baseline: llama-bench's JSON.
"""
import glob, json, math, os, sys

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main(folder):
    runs = []
    for p in glob.glob(os.path.join(folder, "*.json")):
        try:
            j = load(p)
        except Exception:
            continue
        if isinstance(j, dict):
            runs.append((p, j))
    conv = [(p, j) for p, j in runs if j.get("kind", "conversations") == "conversations" and "runs" in j and not (j.get("options") or {}).get("think")]
    # newest run per (mode, conversation)
    best = {}
    for p, j in sorted(conv, key=lambda x: x[1].get("updated", "")):
        for R in j["runs"]:
            for C in R["convs"]:
                best[(R["mode"], C["id"])] = (os.path.basename(p), C)
    order = ["off", "light", "medium", "strong", "ram"]
    modes = sorted({m for m, _ in best}, key=lambda m: order.index(m) if m in order else 99)
    if modes:
        print("Conversation tests (newest run per mode and conversation)")
        print(f"{'mode':8} {'passed':>9} {'checks':>9} {'tok/s':>7}  files")
        answers = {}
        for m in modes:
            P = T = CK = CO = 0
            tok = sec = 0.0
            files = set()
            for (mm, cid), (f, C) in best.items():
                if mm != m:
                    continue
                files.add(f)
                for t in C["turns"]:
                    T += 1; P += bool(t["ok"])
                    CK += len(t["checks"]); CO += sum(bool(c["ok"]) for c in t["checks"])
                    s = t.get("stats") or {}
                    tok += s.get("gen_tokens", 0); sec += s.get("gen_sec", 0)
                    answers[(m, cid, t["i"])] = t["answer"]
            print(f"{m:8} {P:>4}/{T:<4} {CO:>4}/{CK:<4} {tok / sec if sec else 0:>7.2f}  {', '.join(sorted(files))}")
        if "off" in modes:
            for m in modes:
                if m == "off":
                    continue
                keys = [k for k in answers if k[0] == "off"]
                same = sum(1 for k in keys if answers.get((m,) + k[1:]) == answers[k])
                print(f"  {m}: {same} of {len(keys)} answers word for word the same as off")
        print()
    fid = [(p, j) for p, j in runs if j.get("kind") == "fidelity" and j.get("finished")]
    offs = {}
    for p, j in sorted(fid, key=lambda x: x[1].get("updated", "")):
        o = (j.get("modes") or {}).get("off")
        if o:
            offs[os.path.basename(p)] = (o.get("tokens"), o.get("nll"), o.get("agree_ref"))
        print(f"Fidelity test ({os.path.basename(p)}): {j['plan']['answer_tokens']} answer tokens, {j['plan']['prompt_tokens']} prompt tokens")
        print(f"{'mode':8} {'same top':>9} {'KL/token':>9} {'KL>0.1':>7} {'ppl':>8} {'tok/s':>6} {'GB/tok':>7} {'hits':>6} {'swapped':>8}")
        tot = {m: d for m, d in j["modes"].items()}
        ref = tot.get("off")
        for m in j["options"]["modes"]:
            d = tot.get(m)
            if not d:
                continue
            n = d["tokens"]
            st = d.get("stream") or {}
            ppl = "" if m == "off" or not ref else f"{(math.exp((d['nll'] - ref['nll']) / n) - 1) * 100:+.2f}%"
            sw = st.get("swapped", 0) / max(1, st.get("accesses", 1))
            print(f"{m:8} {d['agree_off'] / n * 100:>8.2f}% {d['kl'] / n:>9.4f} {d['kl_big'] / n * 100:>6.2f}% {ppl:>8} {n / d['sec']:>6.2f} "
                  f"{st.get('bytes_read', 0) / n / 1e9:>7.3f} {st.get('hits', 0) / max(1, st.get('accesses', 1)) * 100:>5.1f}% {sw * 100:>7.1f}%")
        print()
    if len(set(offs.values())) == 1 and len(offs) > 1:
        print(f"Exact routing (off) is identical in all {len(offs)} fidelity runs: same tokens, summed NLL and agreement.\n")
    elif len(offs) > 1:
        print("Exact routing (off) differs between fidelity runs:", offs, "\n")
    sp = [(p, j) for p, j in runs if j.get("kind") == "speed" and j.get("rows")]
    if sp:
        p, j = max(sp, key=lambda x: x[1].get("updated", ""))
        tests = j["options"]["tests"]
        print(f"Rafid speed test ({os.path.basename(p)}), tokens/s, mean of {j['options']['reps']} runs")
        print(f"{'mode':8} " + " ".join(f"{t:>8}" for t in tests))
        for m in j["options"]["modes"]:
            cells = []
            for t in tests:
                rs = [r for r in j["rows"] if r["mode"] == m and r["test"] == t and r.get("sec")]
                tok, sec = sum(r["tokens"] for r in rs), sum(r["sec"] for r in rs)
                cells.append(f"{tok / sec:>8.2f}" if sec else f"{'-':>8}")
            print(f"{m:8} " + " ".join(cells))
        print()
    for p in sorted(glob.glob(os.path.join(folder, "baseline-llamacpp-*.json"))):
        if True:
            for r in load(p):
                test = f"tg{r['n_gen']}" if r["n_gen"] else f"pp{r['n_prompt']}"
                print(f"llama.cpp {r['build_number']} {test}: {r['avg_ts']:.2f} ± {r['stddev_ts']:.2f} tok/s "
                      f"(samples {', '.join(f'{x:.2f}' for x in r['samples_ts'])}; ngl {r['n_gpu_layers']}, n_cpu_moe {r['n_cpu_moe']})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/results")
