"""Independent NumPy float64 forward and analytic backward references.

This module does not call the implementation's core, positional helpers, or
model. It deliberately computes in FP64 even when the input case is FP32.
"""

import numpy as np


def _array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _reduce_broadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


def numpy_attention(query, key, value, upstream, *, causal=True, scale=None,
                    attn_mask=None, key_padding_mask=None, positional_bias=None,
                    dropout=0.0, dropout_mask=None):
    """Return (context, analytic gradients, intermediates) for fixed dropout."""
    q, k, v, dy = [np.asarray(_array(t), dtype=np.float64)
                   for t in (query, key, value, upstream)]
    b, h, sq, d = q.shape
    sk = k.shape[2]
    alpha = d ** -0.5 if scale is None else scale
    scores = np.einsum("bhid,bhjd->bhij", q, k) * alpha
    logits = scores.copy()
    if positional_bias is not None:
        logits += _array(positional_bias)
    mask = np.zeros((b, h, sq, sk), dtype=bool)
    if causal:
        mask |= np.arange(sk)[None, :] > np.arange(sq)[:, None]
    if attn_mask is not None:
        supplied = _array(attn_mask)
        if supplied.dtype == np.bool_:
            mask |= supplied
        else:
            logits += supplied
    if key_padding_mask is not None:
        mask |= _array(key_padding_mask)[:, None, None, :]
    logits[mask] = -np.inf
    valid = ~np.isneginf(logits).all(axis=-1, keepdims=True)
    safe = np.where(valid, logits, 0.0)
    exponents = np.exp(safe - np.max(safe, axis=-1, keepdims=True))
    p = exponents / exponents.sum(axis=-1, keepdims=True)
    p = np.where(valid, p, 0.0)
    multiplier = 1.0
    if dropout:
        if dropout_mask is None:
            raise ValueError("the reference requires a fixed mask for nonzero dropout")
        multiplier = _array(dropout_mask).astype(np.float64) / (1 - dropout)
    dropped = p * multiplier
    output = np.einsum("bhij,bhjd->bhid", dropped, v)
    dv = np.einsum("bhij,bhid->bhjd", dropped, dy)
    dp = np.einsum("bhid,bhjd->bhij", dy, v) * multiplier
    dl = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
    dq = alpha * np.einsum("bhij,bhjd->bhid", dl, k)
    dk = alpha * np.einsum("bhij,bhid->bhjd", dl, q)
    gradients = {"query": dq, "key": dk, "value": dv}
    if positional_bias is not None:
        gradients["positional_bias"] = _reduce_broadcast(dl, _array(positional_bias).shape)
    if attn_mask is not None and _array(attn_mask).dtype != np.bool_:
        gradients["attn_mask"] = _reduce_broadcast(dl, _array(attn_mask).shape)
    intermediates = {"scores": scores, "logits": logits, "probabilities": p,
                     "dropped_probabilities": dropped, "context": output}
    return output, gradients, intermediates


def _rotate_numpy(x, positions, rotary_dim, base, *, inverse=False):
    """Separate loop-based RoPE oracle; inverse rotation gives the input gradient."""
    b, _, seq, dim = x.shape
    rd = dim if rotary_dim is None else rotary_dim
    ids = np.arange(seq) if positions is None else _array(positions)
    ids = np.broadcast_to(ids, (b, seq))
    result = x.copy()
    for batch in range(b):
        for token in range(seq):
            for pair in range(rd // 2):
                angle = ids[batch, token] / (base ** (2 * pair / rd))
                cosine, sine = np.cos(angle), np.sin(angle)
                if inverse:
                    sine = -sine
                even, odd = x[batch, :, token, 2 * pair], x[batch, :, token, 2 * pair + 1]
                result[batch, :, token, 2 * pair] = even * cosine - odd * sine
                result[batch, :, token, 2 * pair + 1] = even * sine + odd * cosine
    return result


def numpy_self_attention(x, weights, upstream, config, *, attn_mask=None,
                         key_padding_mask=None, positional_bias=None,
                         position_ids=None, dropout_mask=None):
    """Full projection/RoPE/core/projection oracle, including parameter gradients."""
    x, dy = [np.asarray(_array(t), dtype=np.float64) for t in (x, upstream)]
    state = {name: np.asarray(_array(t), dtype=np.float64) for name, t in weights.items()}
    sbe = config.input_layout == "SBE"
    if sbe:
        x, dy = x.transpose(1, 0, 2), dy.transpose(1, 0, 2)
    batch, seq, embed = x.shape
    heads, dim = config.num_heads, config.head_dim
    wq, wo = state["qkv_proj.weight"], state["out_proj.weight"]
    projected = x @ wq.T
    if config.bias:
        projected += state["qkv_proj.bias"]
    q, k, v = [t.reshape(batch, seq, heads, dim).transpose(0, 2, 1, 3)
               for t in np.split(projected, 3, axis=-1)]
    if config.rope:
        q = _rotate_numpy(q, position_ids, config.rotary_dim, config.rope_base)
        k = _rotate_numpy(k, position_ids, config.rotary_dim, config.rope_base)
    dc = (dy @ wo).reshape(batch, seq, heads, dim).transpose(0, 2, 1, 3)
    context, core_grads, intermediates = numpy_attention(
        q, k, v, dc, causal=config.causal, scale=config.scale, attn_mask=attn_mask,
        key_padding_mask=key_padding_mask, positional_bias=positional_bias,
        dropout=config.dropout, dropout_mask=dropout_mask)
    merged = context.transpose(0, 2, 1, 3).reshape(batch, seq, embed)
    output = merged @ wo.T
    if config.bias:
        output += state["out_proj.bias"]
    dq, dk, dv = [core_grads[name] for name in ("query", "key", "value")]
    if config.rope:
        dq = _rotate_numpy(dq, position_ids, config.rotary_dim, config.rope_base, inverse=True)
        dk = _rotate_numpy(dk, position_ids, config.rotary_dim, config.rope_base, inverse=True)
    dprojected = np.concatenate([t.transpose(0, 2, 1, 3).reshape(batch, seq, embed)
                                 for t in (dq, dk, dv)], axis=-1)
    dx = dprojected @ wq
    gradients = {
        "input": dx.transpose(1, 0, 2) if sbe else dx,
        "qkv_proj.weight": dprojected.reshape(-1, 3 * embed).T @ x.reshape(-1, embed),
        "out_proj.weight": dy.reshape(-1, embed).T @ merged.reshape(-1, embed),
        "query": core_grads["query"], "key": core_grads["key"], "value": core_grads["value"],
    }
    if config.bias:
        gradients["qkv_proj.bias"] = dprojected.sum(axis=(0, 1))
        gradients["out_proj.bias"] = dy.sum(axis=(0, 1))
    for name in ("positional_bias", "attn_mask"):
        if name in core_grads:
            gradients[name] = core_grads[name]
    return output.transpose(1, 0, 2) if sbe else output, gradients, intermediates
