import torch
import torch.nn as nn
import torch.nn.functional as F


def readout_test(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    weight_decay: float = 1e-3,
    optimizer: str = "adam",   # "adam" or "lbfgs"
    lr: float = 1e-3,
    max_iter: int = 100,       # for LBFGS
    epochs: int = 200,         # for Adam
    normalize_in_place: bool = False,
    return_predictions: bool = False,
    return_parameters: bool = False,
):
    """
    Linear probe on fixed train/valid(/test) split, without PCA.

    Pipeline:
    1. Fit z-score normalization on TRAIN only
    2. Train a linear classifier on TRAIN
    3. Report valid/test accuracy

    Assumption:
    - labels are already in [0, C-1]
    """
    assert Xtr.dim() == 2 and Xva.dim() == 2 and Xte.dim() == 2
    assert ytr.dim() == 1 and yva.dim() == 1 and yte.dim() == 1
    assert Xtr.shape[1] == Xva.shape[1]
    assert Xtr.shape[0] == ytr.shape[0]
    assert Xva.shape[0] == yva.shape[0]
    assert Xte.shape[1] == Xtr.shape[1]
    assert Xte.shape[0] == yte.shape[0]

    Xtr = Xtr.detach().to('cpu')
    Xva = Xva.detach().to('cpu')
    Xte = Xte.detach().to('cpu')
    ytr = ytr.detach().to('cpu')
    yva = yva.detach().to('cpu')
    yte = yte.detach().to('cpu')

    device = Xtr.device
    dtype = Xtr.dtype

    # -------------------------
    # 1) Train-only z-score
    # -------------------------
    mu = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)

    if normalize_in_place:
        # Readout callers commonly pass freshly concatenated feature tensors.
        # Reusing those buffers avoids retaining another full train/valid/test
        # feature copy for high-dimensional probes.
        Xtr.sub_(mu).div_(std)
        Xva.sub_(mu).div_(std)
        Xte.sub_(mu).div_(std)
        Xtr_n, Xva_n, Xte_n = Xtr, Xva, Xte
    else:
        Xtr_n = (Xtr - mu) / std
        Xva_n = (Xva - mu) / std
        Xte_n = (Xte - mu) / std

    # -------------------------
    # 2) Build linear classifier
    # -------------------------
    num_classes = int(torch.max(ytr).item()) + 1
    lin = nn.Linear(Xtr_n.shape[1], num_classes, bias=True, device=device, dtype=dtype)

    # -------------------------
    # 3) Train
    # -------------------------
    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(
            lin.parameters(),
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = lin(Xtr_n)
            ce = F.cross_entropy(logits, ytr)
            l2 = 0.5 * weight_decay * sum((p ** 2).sum() for p in lin.parameters())
            loss = ce + l2
            loss.backward()
            return loss

        opt.step(closure)

    elif optimizer.lower() == "adam":
        opt = torch.optim.AdamW(
            lin.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        best_params = None
        best_val_acc = -1.0

        for _ in range(epochs):
            lin.train()
            opt.zero_grad(set_to_none=True)
            logits = lin(Xtr_n)
            loss = F.cross_entropy(logits, ytr)
            loss.backward()
            opt.step()

            lin.eval()
            with torch.no_grad():
                val_pred = lin(Xva_n).argmax(dim=1)
                val_acc = (val_pred == yva).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [param.detach().clone() for param in lin.parameters()]

        if best_params is not None:
            with torch.no_grad():
                for param, best_param in zip(lin.parameters(), best_params):
                    param.copy_(best_param)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # -------------------------
    # 4) Evaluate
    # -------------------------
    readout_score = {}
    with torch.no_grad():
        train_pred = lin(Xtr_n).argmax(dim=1)
        valid_pred = lin(Xva_n).argmax(dim=1)
        test_pred = lin(Xte_n).argmax(dim=1)
        readout_score["train"] = (train_pred == ytr).float().mean().item()
        readout_score["valid"] = (valid_pred == yva).float().mean().item()
        readout_score["test"] = (test_pred == yte).float().mean().item()

    predictions = {
        "train": train_pred.detach().to("cpu"),
        "valid": valid_pred.detach().to("cpu"),
        "test": test_pred.detach().to("cpu"),
    }
    parameters = None
    if return_parameters:
        with torch.no_grad():
            feature_scale = std.squeeze(0)
            feature_mean = mu.squeeze(0)
            effective_weight = lin.weight / feature_scale.unsqueeze(0)
            effective_bias = lin.bias - (
                effective_weight * feature_mean.unsqueeze(0)
            ).sum(dim=1)
            parameters = {
                "weight": effective_weight.detach().to("cpu"),
                "bias": effective_bias.detach().to("cpu"),
            }

    if return_predictions and return_parameters:
        return readout_score, predictions, parameters
    if return_predictions:
        return readout_score, predictions
    if return_parameters:
        return readout_score, parameters

    return readout_score


# def linear_probe_val_with_pca(
#     activity: torch.Tensor,    # [B, D], can be on CUDA
#     labels: torch.Tensor,      # [B], LongTensor on same device
#     target_dim: int = 999999999,     # PCA target dim; if D <= target_dim, PCA is skipped
#     val_ratio: float = 0.2,    # hold-out ratio
#     weight_decay: float = 1e-3,
#     max_iter: int = 50,
#     seed: int = 42
# ) -> float:
#     """
#     Train a linear classifier on a stratified train/val split, using train-only z-score
#     and (optional) PCA to a fixed dimension. Optimizer is LBFGS (full-batch).
#     Returns the validation accuracy (float in [0,1]).

#     Notes:
#     - PCA is fitted on TRAIN features (after standardization) and applied to both TRAIN/VAL.
#     - If the original D <= target_dim, PCA is skipped (identity projection).
#     """
#     assert activity.dim() == 2, "activity must be [B, D]"
#     assert labels.dim() == 1 and labels.shape[0] == activity.shape[0], "labels must be [B]"
#     device, dtype = activity.device, activity.dtype
#     y = labels.view(-1)

#     # -------------------------
#     # 1) Stratified hold-out
#     # -------------------------
#     g = torch.Generator(device='cpu').manual_seed(seed)
#     y_cpu = y.detach().cpu()
#     classes = torch.unique(y_cpu)
#     tr_idx_list, va_idx_list = [], []
#     for c in classes.tolist():
#         idx = torch.where(y_cpu == c)[0]
#         perm = torch.randperm(idx.numel(), generator=g)
#         n_val = max(1, int(round(val_ratio * idx.numel())))
#         va_idx_list.append(idx[perm[:n_val]])
#         tr_idx_list.append(idx[perm[n_val:]])
#     tr_idx = torch.cat(tr_idx_list).to(device)
#     va_idx = torch.cat(va_idx_list).to(device)

#     Xtr = activity.index_select(0, tr_idx)
#     Xva = activity.index_select(0, va_idx)
#     ytr = y.index_select(0, tr_idx)
#     yva = y.index_select(0, va_idx)

#     # -------------------------
#     # 2) Train-only z-score
#     # -------------------------
#     mu = Xtr.mean(dim=0, keepdim=True)
#     std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
#     Xtr = (Xtr - mu) / std
#     Xva = (Xva - mu) / std

#     # -------------------------
#     # 3) PCA to fixed dim (optional)
#     # -------------------------
#     Btr, D = Xtr.shape
#     if D > target_dim:
#         # Fit PCA basis on TRAIN (already centered by z-score, so center=False here)
#         q = min(target_dim, D)
#         # torch.pca_lowrank returns U, S, V such that X ≈ U S V^T
#         U, S, V = torch.pca_lowrank(Xtr, q=q, center=False)
#         Xtr = Xtr @ V        # project to q dims
#         Xva = Xva @ V
#         used_dim = q
#     else:
#         used_dim = D

#     # -------------------------
#     # 4) Linear head + LBFGS
#     # -------------------------
#     num_classes = int(y.max().item() + 1)
#     lin = nn.Linear(used_dim, num_classes, bias=True, device=device, dtype=dtype)
#     opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

#     def closure():
#         opt.zero_grad(set_to_none=True)
#         logits = lin(Xtr)
#         ce = F.cross_entropy(logits, ytr)
#         l2 = 0.5 * weight_decay * sum((p**2).sum() for p in lin.parameters())
#         loss = ce + l2
#         loss.backward()
#         return loss

#     opt.step(closure)

#     # -------------------------
#     # 5) Return validation accuracy
#     # -------------------------
#     with torch.no_grad():
#         pred = lin(Xva).argmax(dim=1)
#         val_acc = (pred == yva).float().mean().item()
#     return float(val_acc)
