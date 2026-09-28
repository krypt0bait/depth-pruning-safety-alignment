"""ReplaceMe: replace a contiguous window of n blocks with a linear transform T,
fitted by least squares and fused into the preceding block's down_proj.

hidden_states[k] is the output of block k-1, so L_i = hidden_states[i + 1].
hidden_states[L] is after the final norm and must never be used as a target.
"""
import torch


def angular_distance(a, b):
    """acos(cos_sim) / pi, as in the official code."""
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return torch.acos((a * b).sum(-1).clamp(-1, 1)) / torch.pi


@torch.no_grad()
def select_cut(model, loader, n):
    """Eq. 4: i* = argmin_i d(L_i, L_{i+n}) on the last non-pad token.
    Blocks i*+1 .. i*+n are removed."""
    L = model.config.num_hidden_layers
    assert 0 < n <= L - 2
    n_cands = L - n - 1                       # keeps i + 1 + n <= L - 1
    dist = torch.zeros(n_cands)
    n_batches = 0
    for batch in loader:
        hs = model(**batch, output_hidden_states=True).hidden_states
        mask = batch["attention_mask"].bool()
        assert mask[:, 0].all(), "expected right padding"
        last = mask.sum(dim=1) - 1
        rows = torch.arange(last.shape[0], device=last.device)
        for i in range(n_cands):
            a = hs[i + 1][rows, last]
            b = hs[i + 1 + n][rows, last]
            dist[i] += angular_distance(a.float(), b.float()).mean().item()
        n_batches += 1
    dist /= n_batches
    return int(dist.argmin()), dist


def fit_transform(model, loader, i_star, n, alpha):
    """Eq. 6:  T = (M^T M + alpha I)^-1 M^T (L_{i+n} - L_i + M),
    where M is block i*'s MLP output (L_i - M is the pre-MLP residual, Eq. 3).
    Accumulated over all non-pad tokens in fp64."""
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    assert i_star + 1 + n <= L - 1

    MtM = torch.zeros(d, d, dtype=torch.float64)
    MtR = torch.zeros(d, d, dtype=torch.float64)
    cache = {}
    hook = model.model.layers[i_star].mlp.register_forward_hook(
        lambda mod, inp, out: cache.__setitem__("M", out))
    try:
        for batch in loader:
            with torch.no_grad():
                hs = model(**batch, output_hidden_states=True).hidden_states
            M = cache.pop("M").float()
            dev = M.device
            R = hs[i_star + 1 + n].to(dev).float() - hs[i_star + 1].to(dev).float() + M
            mask = batch["attention_mask"].bool().to(dev)
            M_, R_ = M[mask].double().cpu(), R[mask].double().cpu()
            MtM += M_.T @ M_
            MtR += M_.T @ R_
    finally:
        hook.remove()
    return torch.linalg.solve(MtM + alpha * torch.eye(d, dtype=torch.float64), MtR)


def fuse_transform(model, i_star, T):
    """M T = h (T^T W)^T, so W <- T^T W. Requires no down_proj bias."""
    down = model.model.layers[i_star].mlp.down_proj
    assert down.bias is None
    W = down.weight
    W.data = (T.to(W.device, torch.float32).T @ W.data.float()).to(W.dtype)


@torch.no_grad()
def check_fusion(model, i_star, T, batch):
    """Fuse T in place and assert the MLP now outputs M_old @ T."""
    cache = {}
    hook = model.model.layers[i_star].mlp.register_forward_hook(
        lambda mod, inp, out: cache.__setitem__("M", out.detach().clone()))
    try:
        model(**batch)
        M_old = cache.pop("M")
        fuse_transform(model, i_star, T)
        model(**batch)
        M_new = cache.pop("M")
    finally:
        hook.remove()
    expected = M_old.float() @ T.to(M_old.device).float()
    rel_err = ((M_new.float() - expected).abs().max() / (expected.abs().max() + 1e-12)).item()
    assert rel_err < 1e-2, f"fusion mismatch: rel_err={rel_err:.3e}"
