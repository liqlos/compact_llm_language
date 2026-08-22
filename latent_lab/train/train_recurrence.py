"""Tiny recurrence trainer — plumbing proof at toy scale (T1 analog).

Trains a small torch module where answers must flow through K latent steps:
embedding -> prefix -> K x interval-block -> head. Proves the loss decreases
and that a causal ablation (bypassing the latent loop) HURTS accuracy —
i.e., the model actually uses the loop. This is NOT a language-model result
and never substitutes for Qwen experiments; it validates data->loss->forward
so the GPU configuration for real models is trustworthy.

Run:  python -m latent_lab.train.train_recurrence --steps 300
"""

from __future__ import annotations

import argparse
import json


def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "torch required for training; install group `lab`"
        ) from e
    return torch


def build_model(vocab: int = 128, dim: int = 32):
    torch = _torch()
    import torch.nn.functional as F
    from torch import nn

    class IntervalBlock(nn.Module):
        """Stand-in for a localized recurrent interval."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)
            self.ln = nn.LayerNorm(dim)

        def forward(self, h):
            return h + self.ln(F.gelu(self.fc2(F.gelu(self.fc1(h)))))

    class TinyRecurrence(nn.Module):
        """Answers are decoded ONLY from post-recurrence state."""

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.prefix = nn.Linear(dim, dim)
            self.block = IntervalBlock()
            self.head = nn.Linear(dim, vocab)

        def forward(self, ids, k_steps: int, ablate_latent: bool = False):
            h = self.prefix(self.emb(ids))
            lat = h.mean(dim=1, keepdim=True)      # initial latent state
            if ablate_latent:
                lat = torch.zeros_like(lat)        # causal ablation
            else:
                for _ in range(k_steps):           # continuous loop,
                    lat = self.block(lat)          # no decode inside
            h = h + lat
            return self.head(h[:, -1:, :]), lat

    return TinyRecurrence()


def run_smoke_training(steps: int = 300, lr: float = 3e-3, seed: int = 0,
                       k_steps: int = 4) -> dict:
    from .data import make_chain_dataset
    from .losses import answer_and_align_loss

    torch = _torch()
    torch.manual_seed(seed)
    data = make_chain_dataset(n=64)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        ex = data[step % len(data)]
        ids = torch.tensor([ex.input_ids])
        logits, lat = model(ids, k_steps=k_steps)
        total, ans, _align = answer_and_align_loss(logits, ex.answer_ids,
                                                   student_readout=None,
                                                   teacher_repr=None)
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(float(ans.detach()))

    def accuracy(ablated: bool) -> float:
        correct = 0
        with torch.no_grad():
            for ex in data:
                ids = torch.tensor([ex.input_ids])
                logits, _ = model(ids, k_steps=k_steps, ablate_latent=ablated)
                pred = int(logits[0, -1].argmax())
                correct += int(pred == ex.answer_ids[0])
        return correct / len(data)

    acc_normal = accuracy(ablated=False)
    acc_ablated = accuracy(ablated=True)
    return {
        "steps": steps,
        "k_steps": k_steps,
        "initial_answer_loss": losses[0],
        "final_answer_loss": losses[-1],
        "loss_decreased": bool(losses[-1] < losses[0]),
        "train_accuracy_normal": acc_normal,
        "train_accuracy_ablated": acc_ablated,
        # anti-cheat: if ablating the loop does not hurt, the loop is unused
        "latent_path_causally_used": bool(
            losses[-1] < losses[0] and acc_normal > 0.9
            and acc_ablated < acc_normal - 0.05
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = run_smoke_training(steps=args.steps)
    print(json.dumps(rep, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    ok = rep["loss_decreased"] and rep["latent_path_causally_used"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
