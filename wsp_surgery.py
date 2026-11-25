"""Pure zero-gradient Weight-Space Programming (WSP) surgery primitives.

Operates directly on Hugging Face Safetensors checkpoints (supports both
Qwen3_5ForConditionalGeneration and standard CausalLM layouts) with exact
parameter-modification accounting.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


def get_language_layers(model: Any) -> Tuple[Any, str]:
    """Locate decoder layers and their state-dict key prefix across Qwen3.5 / Qwen3."""
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers, "model.language_model.layers"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.layers"
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers, "language_model.layers"
    raise RuntimeError(f"Could not locate decoder layers on model class {type(model).__name__}")


def _find_down_proj_key(tensor_keys: List[str], layer_idx: int) -> str:
    """Resolve exact safetensors key for layer_idx's MLP down_proj.weight."""
    suffix = f"layers.{layer_idx}.mlp.down_proj.weight"
    matches = [
        k
        for k in tensor_keys
        if k.endswith(suffix)
        and "visual" not in k
        and "vision" not in k
        and "audio" not in k
    ]
    if not matches:
        raise KeyError(f"Could not find tensor ending with '{suffix}' in checkpoint keys.")
    # Prefer language_model key if multiple
    matches.sort(key=lambda k: (0 if "language_model" in k else 1, len(k)))
    return matches[0]


def compute_patched_weight(
    weight: torch.Tensor,
    diag_layer: Dict[str, Any],
    mode: str,
    alpha: float,
    top_k: int = 64,
    rank: int = 1,
    rank_by: str = "act_weighted",
    vec_source: str = "resid_loop_dir",
    think_weight: float = 0.5,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """Apply a zero-gradient weight-space edit to a single `down_proj.weight` [d_model, d_ff].

    Returns (new_weight_same_dtype, num_modified_elements, stats).
    """
    orig_dtype = weight.dtype
    W = weight.detach().to(torch.float32).clone()  # [d_model, d_ff]
    d_model, d_ff = W.shape

    if vec_source in diag_layer:
        v_mean = diag_layer[vec_source].to(torch.float32)
    else:
        v_mean = diag_layer["loop_dir"].to(torch.float32)
    v_mean = v_mean / (v_mean.norm() + 1e-12)

    pca_key = "resid_pca_dirs" if (vec_source == "resid_loop_dir" and "resid_pca_dirs" in diag_layer) else "pca_dirs"
    pca_dirs = diag_layer[pca_key].to(torch.float32)  # [r_max, d_model], orthonormal rows

    if rank_by == "resid_act_weighted" and "col_scores_resid_act_weighted" in diag_layer:
        scores = diag_layer["col_scores_resid_act_weighted"].to(torch.float32)
    elif rank_by == "z_excess" and "col_scores_z_excess" in diag_layer:
        scores = diag_layer["col_scores_z_excess"].to(torch.float32)
    elif rank_by == "z_loop_abs" and "z_loop_mean" in diag_layer:
        scores = diag_layer["z_loop_mean"].to(torch.float32).abs()
    elif rank_by == "z_diff" and "col_scores_z_diff" in diag_layer:
        scores = diag_layer["col_scores_z_diff"].to(torch.float32)
    elif rank_by == "act_weighted" and "col_scores_act_weighted" in diag_layer:
        scores = diag_layer["col_scores_act_weighted"].to(torch.float32)
    elif rank_by == "abs_act_weighted" and "col_scores_act_weighted" in diag_layer:
        scores = diag_layer["col_scores_act_weighted"].to(torch.float32).abs()
    elif rank_by == "weight_aligned":
        scores = (v_mean @ W)  # [d_ff]
    elif rank_by == "abs_weight_aligned":
        scores = (v_mean @ W).abs()
    else:
        scores = (v_mean @ W)

    k = min(int(top_k), d_ff)
    top_idx = torch.topk(scores, k=k, largest=True).indices

    # Basis U: [d_model, r]
    r = max(1, min(int(rank), pca_dirs.shape[0] + 1))
    if r == 1:
        U = v_mean.unsqueeze(1)
    else:
        basis = [v_mean]
        for row in pca_dirs:
            u = row.clone()
            for b in basis:
                u = u - torch.dot(u, b) * b
            n = u.norm()
            if n > 1e-6:
                basis.append(u / n)
            if len(basis) >= r:
                break
        U = torch.stack(basis, dim=1)

    W_orig = W.clone()
    if mode == "col_ablate":
        # Project out U subspace strictly on the top-K culprit columns (MIT CSAIL style!)
        sub = W[:, top_idx]  # [d_model, k]
        proj = U @ (U.T @ sub)
        W[:, top_idx] = sub - float(alpha) * proj
        modified_elems = int(d_model * k)
    elif mode == "col_steer":
        # Direct conditional anti-loop steering: subtract alpha * col_norm * v_mean from top-K loop-firing cols
        sub = W[:, top_idx]  # [d_model, k]
        col_norms = sub.norm(dim=0, keepdim=True)  # [1, k]
        W[:, top_idx] = sub - float(alpha) * col_norms * v_mean.unsqueeze(1)
        modified_elems = int(d_model * k)
    elif mode == "col_steer_think":
        # Conditional anti-loop + </think> boost on top-K loop-firing cols
        sub = W[:, top_idx]
        col_norms = sub.norm(dim=0, keepdim=True)
        steer = -v_mean
        if "think_end_dir" in diag_layer:
            t_dir = diag_layer["think_end_dir"].to(torch.float32)
            t_dir = t_dir / (t_dir.norm() + 1e-12)
            steer = steer + float(think_weight) * t_dir
            steer = steer / (steer.norm() + 1e-12)
        W[:, top_idx] = sub + float(alpha) * col_norms * steer.unsqueeze(1)
        modified_elems = int(d_model * k)
    elif mode == "col_steer_signed":
        # Sign-aligned constructive conditional steering: multiplies by sign(z_loop_mean[j])
        # so ALL top-K SwiGLU loop-firing neurons constructively push along (-v_mean + tw * think_end_dir)!
        sub = W[:, top_idx]
        col_norms = sub.norm(dim=0, keepdim=True)
        steer = -v_mean
        if "think_end_dir" in diag_layer and float(think_weight) != 0.0:
            t_dir = diag_layer["think_end_dir"].to(torch.float32)
            t_dir = t_dir / (t_dir.norm() + 1e-12)
            steer = steer + float(think_weight) * t_dir
            steer = steer / (steer.norm() + 1e-12)
        if "z_loop_mean" in diag_layer:
            z_signs = torch.sign(diag_layer["z_loop_mean"].to(torch.float32)[top_idx])
            z_signs = torch.where(z_signs == 0, torch.ones_like(z_signs), z_signs).unsqueeze(0)
        else:
            z_signs = torch.ones((1, k), dtype=torch.float32)
        W[:, top_idx] = sub + float(alpha) * col_norms * z_signs * steer.unsqueeze(1)
        modified_elems = int(d_model * k)
    elif mode == "col_ablate_signed":
        sub = W[:, top_idx]
        dots = (v_mean @ sub).clamp(min=0.0)
        W[:, top_idx] = sub - float(alpha) * v_mean.unsqueeze(1) * dots.unsqueeze(0)
        modified_elems = int(d_model * k)
    elif mode == "col_damp":
        W[:, top_idx] = W[:, top_idx] * (1.0 - float(alpha))
        modified_elems = int(d_model * k)
    elif mode == "subspace_ablate":
        proj = U @ (U.T @ W)
        W = W - float(alpha) * proj
        modified_elems = int(d_model * d_ff)
    elif mode == "col_ablate_dual":
        sub = W[:, top_idx]
        proj = U @ (U.T @ sub)
        W[:, top_idx] = sub - float(alpha) * proj
        if "term_dir" in diag_layer:
            t_dir = diag_layer["term_dir"].to(torch.float32)
            t_dir = t_dir - U @ (U.T @ t_dir)
            tn = t_dir.norm()
            if tn > 1e-6:
                t_dir = t_dir / tn
                term_scores = (t_dir @ W_orig)
                term_idx = torch.topk(term_scores, k=min(k // 2, d_ff), largest=True).indices
                t_sub = W[:, term_idx]
                t_dots = (t_dir @ t_sub).clamp(min=0.0)
                W[:, term_idx] = t_sub + (0.25 * float(alpha)) * t_dir.unsqueeze(1) * t_dots.unsqueeze(0)
                modified_elems = int(d_model * len(set(top_idx.tolist()) | set(term_idx.tolist())))
            else:
                modified_elems = int(d_model * k)
        else:
            modified_elems = int(d_model * k)
    else:
        raise ValueError(f"Unknown WSP mode: {mode}")

    delta_norm = float((W - W_orig).norm().item())
    orig_norm = float(W_orig.norm().item())
    stats = {
        "modified_elems": modified_elems,
        "selected_cols": top_idx.tolist()[:16],
        "num_selected_cols": int(k) if mode != "subspace_ablate" else int(d_ff),
        "delta_l2": round(delta_norm, 6),
        "rel_delta": round(delta_norm / (orig_norm + 1e-12), 6),
    }
    return W.to(orig_dtype), modified_elems, stats


def apply_patch_to_checkpoint(
    base_dir: str,
    out_dir: str,
    patch_spec: Dict[str, Any],
    diagnosis: Dict[str, Any],
    copy_files: bool = False,
) -> Dict[str, Any]:
    """Create a patched HF checkpoint directory from base_dir using zero-gradient WSP.

    Uses symlinks for unmodified files when `copy_files=False` (fast temp eval),
    or full copies when `copy_files=True` (clean exportable Hugging Face repo).
    """
    src = Path(base_dir)
    dst = Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Identify safetensors layout (single file vs sharded)
    index_path = src / "model.safetensors.index.json"
    single_path = src / "model.safetensors"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Dict[str, str] = index_data["weight_map"]
        all_keys = list(weight_map.keys())
    elif single_path.exists():
        tensors_single = load_file(str(single_path))
        all_keys = list(tensors_single.keys())
        weight_map = {k: "model.safetensors" for k in all_keys}
    else:
        raise FileNotFoundError(f"No model.safetensors or index.json found in {src}")

    layers: List[int] = [int(x) for x in patch_spec["layers"]]
    mode: str = str(patch_spec.get("mode", "col_ablate"))
    alpha: float = float(patch_spec.get("alpha", 1.0))
    top_k: int = int(patch_spec.get("top_k", 64))
    rank: int = int(patch_spec.get("rank", 1))
    rank_by: str = str(patch_spec.get("rank_by", "act_weighted"))
    vec_source: str = str(patch_spec.get("vec_source", "resid_loop_dir"))
    think_weight: float = float(patch_spec.get("think_weight", 0.5))

    # Group targeted keys by shard filename
    shard_to_targets: Dict[str, List[Tuple[int, str]]] = {}
    for l_idx in layers:
        key = _find_down_proj_key(all_keys, l_idx)
        shard = weight_map[key]
        shard_to_targets.setdefault(shard, []).append((l_idx, key))

    # Copy or symlink all non-modified files
    for entry in src.iterdir():
        if entry.name.startswith(".") or entry.is_dir():
            continue
        if entry.name in shard_to_targets:
            continue
        target_path = dst / entry.name
        if copy_files:
            shutil.copy2(entry, target_path)
        else:
            os.symlink(entry.resolve(), target_path)

    # Modify only the targeted safetensors shard(s)
    total_modified = 0
    layer_stats: Dict[str, Any] = {}
    total_params = int(diagnosis.get("total_params", 0))

    for shard_name, targets in shard_to_targets.items():
        tensors = load_file(str(src / shard_name))
        if total_params == 0 and not index_path.exists():
            total_params = sum(int(t.numel()) for t in tensors.values())
        for l_idx, key in targets:
            diag_layer = diagnosis["layers"][l_idx]
            new_w, mod_elems, st = compute_patched_weight(
                tensors[key],
                diag_layer=diag_layer,
                mode=mode,
                alpha=alpha,
                top_k=top_k,
                rank=rank,
                rank_by=rank_by,
                vec_source=vec_source,
                think_weight=think_weight,
            )
            tensors[key] = new_w
            total_modified += mod_elems
            layer_stats[str(l_idx)] = {"tensor_key": key, **st}
        save_file(tensors, str(dst / shard_name), metadata={"format": "pt"})

    pct_modified = (100.0 * total_modified / total_params) if total_params > 0 else 0.0
    summary = {
        "patch_spec": patch_spec,
        "total_params": total_params,
        "modified_params": total_modified,
        "pct_modified": round(pct_modified, 6),
        "layer_stats": layer_stats,
    }
    (dst / "wsp_patch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
