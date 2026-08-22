"""Losses for latent-recurrence training.

- answer loss: cross-entropy on answer tokens given post-recurrence state
  (primary; the task must be solved THROUGH the latent loop)
- alignment loss (CODI-like, secondary): MSE between student readout slots
  and a projected teacher representation. Slots are NOT forced to match
  individual words.
"""

from __future__ import annotations


def answer_and_align_loss(student_logits, answer_ids, *,
                          student_readout=None, teacher_repr=None,
                          align_weight: float = 0.2):
    """torch tensors in; total, answer_loss, align_loss out.

    Guarded torch import: this module is only used inside training runs.
    """
    import torch
    import torch.nn.functional as F

    logits = student_logits[:, -len(answer_ids):, :]
    target = torch.tensor(answer_ids, device=logits.device).unsqueeze(0).expand(
        logits.shape[0], -1
    )
    ans = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          target.reshape(-1))
    align = torch.tensor(0.0, device=logits.device)
    if student_readout is not None and teacher_repr is not None:
        if student_readout.shape == teacher_repr.shape:
            align = F.mse_loss(student_readout, teacher_repr)
        else:
            # project teacher to student dim if needed
            proj = torch.nn.Linear(teacher_repr.shape[-1],
                                   student_readout.shape[-1],
                                   device=teacher_repr.device,
                                   dtype=teacher_repr.dtype)
            align = F.mse_loss(student_readout, proj(teacher_repr))
    return ans + align_weight * align, ans, align
