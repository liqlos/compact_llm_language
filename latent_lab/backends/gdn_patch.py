"""Autograd-safe rewrites of the pure-torch Qwen3.5 GDN kernels.

transformers' torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule
accumulate outputs with in-place indexed writes; on some backends (MPS) the
saved-tensor version check rejects the resulting graph, making gradient
flow through GDN layers impossible. These mirrors keep the math identical
but accumulate out-of-place, so LoRA gradients can traverse the interval
layers during latent-recurrence training.

Install once before training:
    from latent_lab.backends.gdn_patch import install; install()
Idempotent; affects only new calls.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_INSTALLED = False


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule_safe(
    query, key, value, g, beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query)
        key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)]
    bsz, nh, seq_len, khd = key.shape
    vhd = value.shape[-1]
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale

    state = (torch.zeros(bsz, nh, khd, vhd, dtype=value.dtype,
                         device=value.device) if initial_state is None
             else initial_state.to(value))
    outs = []
    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=2) if outs else \
        torch.zeros(bsz, nh, 0, vhd, dtype=value.dtype, device=value.device)
    final = state if output_final_state else None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, final


def _triangularize_out_of_place(attn, chunk_size):
    """Functional mirror of the in-place row-update loop."""
    rows = []
    for i in range(chunk_size):
        if i == 0:
            rows.append(attn[..., 0, :])
            continue
        top = torch.stack(rows[:i], dim=-2)              # [..., i, cs]
        row = attn[..., i, :i]
        upd = row + (row.unsqueeze(-1) * top[..., :i, :i]).sum(-2)
        rows.append(torch.cat([upd, attn[..., i, i:]], dim=-1))
    return torch.stack(rows, dim=-2)


def torch_chunk_gated_delta_rule_safe(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query)
        key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)]
    bsz, nh, seq_len, khd = key.shape
    vhd = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total = seq_len + pad
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=0)
    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp()
                  .float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    attn = _triangularize_out_of_place(attn, chunk_size)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (torch.zeros(bsz, nh, khd, vhd, dtype=value.dtype,
                         device=value.device) if initial_state is None
             else initial_state.to(value))
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=1)
    chunk_outs = []
    n_chunks = total // chunk_size
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        chunk_outs.append(attn_inter + a_i @ v_new)
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i])
                    .exp()[..., None]).transpose(-1, -2) @ v_new)

    out = torch.stack(chunk_outs, dim=2) if chunk_outs else None
    final = state if output_final_state else None
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])
    out = out[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, final


def install() -> None:
    global _INSTALLED
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    from transformers.cache_utils import LinearAttentionLayer

    mq.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule_safe
    mq.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule_safe

    if not getattr(LinearAttentionLayer, "_rcc_rebind_patched", False):
        def rec_rebind(self, recurrent_states, state_idx=0, **kwargs):
            # rebind a detached clone instead of copy_ into a buffer that
            # lazy-initialization may have aliased to a graph tensor
            self.recurrent_states[state_idx] = recurrent_states.detach().clone()
            return self.recurrent_states[state_idx]

        def conv_rebind(self, conv_states, state_idx=0,
                        conv_kernel_size=None, **kwargs):
            full = conv_states
            if self.is_conv_states_initialized[state_idx] and \
                    self.has_previous_state[state_idx]:
                full = torch.cat([self.conv_states[state_idx], conv_states],
                                 dim=-1)
            keep = (conv_kernel_size or
                    (self.conv_kernel_size[state_idx]
                     if self.is_conv_states_initialized[state_idx]
                     else full.shape[-1]))
            self.conv_states[state_idx] = full[..., -keep:].detach().clone()
            self.has_previous_state[state_idx] = True
            self.is_conv_states_initialized[state_idx] = True
            return full

        LinearAttentionLayer.update_recurrent_state = rec_rebind
        LinearAttentionLayer.update_conv_state = conv_rebind
        LinearAttentionLayer._rcc_rebind_patched = True
    _INSTALLED = True
