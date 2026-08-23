"""Textual baselines A/B/C on Qwen3.5-2B over the behavioral suite v2.

  A  direct / no-thinking   (raw completion prompting, matches latent input)
  B  native thinking        (chat template, enable_thinking=True)
  C  capped thinking        (same as B, hard max_new_tokens budget)

Every run records: model revision, suite sha, sampling params, seed,
accuracy, NON_TERMINATION count, generated tokens, wall-clock, prefill vs
decode split, peak memory. Greedy decoding everywhere (reproducibility).

Usage:
  python -m latent_lab.bench.text_baselines --mode A --split test_id \
      --out .rcc_work/text_A_test_id.json [--n-per-family 16] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import resource
import time

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

SYSTEM_THINK = ("You are a precise solver. Work out the answer step by "
                "step, then finish with a final line 'Answer: X' where X is "
                "the answer only.")
ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def canonical(text: str) -> str:
    t = text.strip().strip(".,;:!?\"'`").lower()
    return t


SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def strip_special(text: str) -> str:
    """Cut everything from the first special token onward."""
    for tok in SPECIAL_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    return text


def parse_answer(generated: str) -> tuple[str | None, str]:
    """Returns (parsed_answer_or_None, status)."""
    generated = strip_special(generated)
    matches = ANSWER_RE.findall(generated)
    if not matches:
        return None, "no_answer_marker"
    return canonical(matches[-1]), "ok"


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


SYSTEM_DIRECT = ("Answer the question directly and exactly. End your reply "
                 "with a final line 'Answer: X' where X is the answer only.")


def build_prompt(tok, ex, mode: str):
    """Returns token ids for the example under baseline mode."""
    if mode == "A":
        sys_msg = SYSTEM_DIRECT
        thinking = False
    else:
        sys_msg = SYSTEM_THINK
        thinking = True
    user = ex.prompt.rsplit("Answer:", 1)[0].strip()
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
        enable_thinking=thinking)
    return ids.input_ids


def run_batched(model, tok, prompts_ids, max_new, device):
    """Left-padded batched greedy generation; returns list of generated strs."""
    import torch

    pad_id = tok.pad_token_id or tok.eos_token_id
    maxlen = max(x.shape[1] for x in prompts_ids)
    input_ids, mask = [], []
    for x in prompts_ids:
        n = maxlen - x.shape[1]
        input_ids.append(torch.cat([
            torch.full((1, n), pad_id, dtype=x.dtype), x], dim=1))
        mask.append(torch.cat([torch.zeros((1, n), dtype=torch.long),
                               torch.ones((1, x.shape[1]), dtype=torch.long)],
                              dim=1))
    input_ids = torch.cat(input_ids).to(device)
    am = torch.cat(mask).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=am,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad_id)
    gen, n_news = [], []
    for i in range(len(prompts_ids)):
        gen.append(tok.decode(out[i, input_ids.shape[1]:],
                              skip_special_tokens=False))
        row = out[i, input_ids.shape[1]:]
        pad_pos = (row == pad_id).nonzero()
        n_news.append(int(pad_pos[0].item()) if pad_pos.numel()
                      else int(row.shape[0]))
    return gen, n_news


def score_example(generated: str, ex) -> dict:
    terminated = "<|im_end|>" in generated or "<|endoftext|>" in generated
    generated = strip_special(generated)
    ans, status = parse_answer(generated)
    correct = 0.0
    if ans is None:
        # accept a bare final line only when it exactly names a candidate
        tail = [ln for ln in generated.strip().splitlines() if ln.strip()]
        cands = [canonical(c) for c in ex.candidates]
        if tail and canonical(tail[-1]) in cands:
            ans = canonical(tail[-1])
            status = "last_line"
    if not terminated:
        # silent truncation is NEVER a correct answer
        status = "NON_TERMINATION"
    elif ans is not None:
        correct = 1.0 if ans == canonical(ex.answer) else 0.0
    return {"status": status, "correct": correct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C"], required=True)
    ap.add_argument("--split", default="test_id",
                    choices=["validation", "test_id", "test_ood"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total examples (smoke runs)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    args = ap.parse_args()
    MODEL_ID, REVISION = args.model, args.revision
    globals()["MODEL_ID"], globals()["REVISION"] = MODEL_ID, REVISION

    import torch
    import transformers

    from latent_lab.bench.suite import build_suite

    defaults = {"A": 512, "B": 512, "C": 128}
    max_new = args.max_new_tokens or defaults[args.mode]

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID,
                                                     revision=REVISION)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    model.to(args.device)

    suite = build_suite()
    exs = list(getattr(suite, args.split))
    if args.limit:
        exs = exs[: args.limit]

    records = []
    t_run0 = time.perf_counter()
    bs = args.batch
    for start in range(0, len(exs), bs):
        chunk = exs[start: start + bs]
        ids_list = [build_prompt(tok, ex, args.mode) for ex in chunk]
        lens = [x.shape[1] for x in ids_list]
        t0 = time.perf_counter()
        gens, n_news = run_batched(model, tok, ids_list, max_new, args.device)
        dt = time.perf_counter() - t0
        for ex, ids, n_in, gen, n_new in zip(chunk, ids_list, lens, gens,
                                             n_news):
            sc = score_example(gen, ex)
            records.append({
                "ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
                "expected": ex.answer, "status": sc["status"],
                "correct": sc["correct"],
                "n_generated": n_new,
                "prompt_tokens": n_in,
                "seconds": round(dt / len(chunk), 3),
                "generated_preview": gen[:400],
            })
        done = len(records)
        if done % (bs * 10) < bs or done == len(exs):
            acc_now = sum(r["correct"] for r in records) / len(records)
            print(f"[{args.mode}/{args.split}] {done}/{len(exs)} "
                  f"acc={acc_now:.3f}", flush=True)

    wall = time.perf_counter() - t_run0
    acc = sum(r["correct"] for r in records) / len(records)
    nonterm = sum(1 for r in records if r["status"] == "NON_TERMINATION")
    by_family = {}
    for f in sorted({r["family"] for r in records}):
        rs = [r for r in records if r["family"] == f]
        by_family[f] = {
            "n": len(rs),
            "acc": round(sum(x["correct"] for x in rs) / len(rs), 4),
            "mean_gen_tokens": round(sum(x["n_generated"] for x in rs) /
                                     len(rs), 1),
        }
    depths = sorted({r["depth"] for r in records})
    by_depth = {str(d): round(sum(r["correct"] for r in records
                                  if r["depth"] == d) /
                             max(1, sum(1 for r in records
                                        if r["depth"] == d)), 4)
                for d in depths}
    gpu = {}
    if args.device.startswith("cuda"):
        import torch
        gpu = {"cuda_peak_alloc_mib": round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 1)}
    report = {
        "gpu_mem": gpu,
        "baseline": args.mode,
        "mode_desc": {"A": "direct raw-completion, no thinking",
                      "B": "native chat thinking, cap " + str(max_new),
                      "C": "native chat thinking, hard cap " + str(max_new)}[
            args.mode],
        "model": MODEL_ID, "revision": REVISION,
        "sampling": {"do_sample": False, "temperature": 0,
                     "max_new_tokens": max_new},
        "seed": "greedy(n/a)", "split": args.split, "suite_sha256":
            suite.manifest()["sha256"],
        "platform": platform.platform(),
        "dtype": "bfloat16", "device": args.device,
        "n_examples": len(records), "accuracy": round(acc, 4),
        "non_termination_count": nonterm,
        "wall_seconds": round(wall, 1), "by_family": by_family,
        "by_depth": by_depth, "peak_rss_mib": round(peak_rss_mib(), 1),
        "records": records,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"== {args.mode}/{args.split}: acc={report['accuracy']} "
          f"nonterm={nonterm} wall={report['wall_seconds']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
