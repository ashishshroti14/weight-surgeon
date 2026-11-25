"""Budget-guarded Weight-Space Programming (WSP) pipeline on Modal.

Strict $50 budget discipline (hard software stop at $45.00 cumulative spend).
Runs on Modal GPUs with Modal Volume v2 caching.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "wsp-weight-space-programming"
VOLUME_NAME = "wsp-research-vol"
VOL_PATH = "/vol"
SPEND_LOG = f"{VOL_PATH}/spend_log.jsonl"
HARD_BUDGET_CAP_USD = 45.0  # Safety margin under user's $50 strict budget
GPU_TYPE = "L40S"  # 48 GB VRAM ($2.17/hr all-in); fast batched vLLM & PyTorch hooks for 0.8B-4B models

# All-in Modal Starter rates (GPU + 2 CPU + 16 GiB RAM = +$0.22/hr)
GPU_RATES_PER_HR = {
    "L4": 1.02,
    "A10G": 1.32,
    "L40S": 2.17,
    "A100-80GB": 2.72,
    "H100": 4.17,
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "langdetect",
        "einops",
        "pillow",
        "qwen-vl-utils",
    )
    .env(
        {
            "HF_HOME": f"{VOL_PATH}/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_file("eval_utils.py", "/root/eval_utils.py")
    .add_local_file("wsp_surgery.py", "/root/wsp_surgery.py")
)


def _read_total_spend() -> float:
    p = Path(SPEND_LOG)
    if not p.exists():
        return 0.0
    total = 0.0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += float(json.loads(line).get("cost_usd", 0.0))
        except Exception:
            pass
    return total


def _guard_budget(stage: str) -> float:
    vol.reload()
    spent = _read_total_spend()
    if spent >= HARD_BUDGET_CAP_USD:
        raise RuntimeError(
            f"BUDGET GUARD ABORT at stage '{stage}': spent ${spent:.2f} >= cap ${HARD_BUDGET_CAP_USD:.2f}"
        )
    return spent


def _record_spend(stage: str, gpu: str, elapsed_sec: float, meta: Optional[Dict[str, Any]] = None) -> float:
    rate = GPU_RATES_PER_HR.get(gpu, 2.17)
    cost = (elapsed_sec / 3600.0) * rate
    entry = {
        "ts": time.time(),
        "stage": stage,
        "gpu": gpu,
        "elapsed_sec": round(elapsed_sec, 2),
        "cost_usd": round(cost, 4),
        "meta": meta or {},
    }
    Path(SPEND_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    vol.commit()
    return cost


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def _ensure_local_snapshot(model_id: str) -> str:
    """Download/cache model files once on Volume and return local directory path."""
    from huggingface_hub import snapshot_download

    target_dir = f"{VOL_PATH}/models/{_slug(model_id)}"
    if Path(target_dir, "config.json").exists():
        return target_dir
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        ignore_patterns=["*.gguf", "*.bin", "original/*"],
    )
    vol.commit()
    return target_dir


def _build_math_prompt(tokenizer: Any, question: str, enable_thinking: bool = True) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{question.strip()}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_plain_prompt(tokenizer: Any, prompt: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": prompt.strip()}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _load_benchmarks() -> Dict[str, Any]:
    """Load disjoint practice, val, and held-out test benchmarks cleanly."""
    from datasets import load_dataset
    from eval_utils import extract_gsm8k_gold

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    math500_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math500_set = {row["problem"].strip() for row in math500_test}

    # Competition MATH train excluding any overlap with MATH-500 test
    comp_math = load_dataset("qwedsacf/competition_math", split="train")
    math_pool = [
        {"problem": r["problem"], "answer": r["solution"]}
        for r in comp_math
        if r["problem"].strip() not in math500_set
    ]
    # Extract boxed gold for competition_math
    from eval_utils import extract_boxed

    math_clean = []
    for item in math_pool:
        g = extract_boxed(item["answer"])
        if g:
            math_clean.append({"problem": item["problem"], "answer": g})

    ifeval = load_dataset("google/IFEval", split="train")

    return {
        "gsm_practice": [
            {"problem": r["question"], "answer": extract_gsm8k_gold(r["answer"])}
            for r in gsm_train.select(range(0, 80))
        ],
        "gsm_val": [
            {"problem": r["question"], "answer": extract_gsm8k_gold(r["answer"])}
            for r in gsm_train.select(range(80, 140))
        ],
        "gsm_test": [
            {"problem": r["question"], "answer": extract_gsm8k_gold(r["answer"])}
            for r in gsm_test
        ],
        "math_practice": math_clean[:80],
        "math_val": math_clean[80:140],
        "math_test": [
            {"problem": r["problem"], "answer": r["answer"]}
            for r in math500_test
        ],
        "ifeval_test": [
            {
                "prompt": r["prompt"],
                "instruction_id_list": r["instruction_id_list"],
                "kwargs": r["kwargs"],
            }
            for r in ifeval
        ],
    }


def _run_vllm_eval_on_model(
    model_path: str,
    gsm_items: List[Dict[str, str]],
    math_items: List[Dict[str, str]],
    ifeval_items: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
) -> Dict[str, Any]:
    """Run deterministic greedy vLLM evaluation (`presence_penalty=0.0`) on math & optional IFEval."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from eval_utils import (
        detect_repetition_loop,
        extract_math_prediction,
        grade_ifeval_sample,
        math_answers_equal,
        strip_thinking,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=max(max_tokens + 2048, 8192),
        trust_remote_code=True,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
    )

    def _eval_math_suite(items: List[Dict[str, str]]) -> Dict[str, Any]:
        if not items:
            return {}
        prompts = [_build_math_prompt(tokenizer, it["problem"], enable_thinking) for it in items]
        outputs = llm.generate(prompts, sampling)
        correct = 0
        truncated = 0
        looping = 0
        closed_think = 0
        total_toks = 0
        for item, p_str, out in zip(items, prompts, outputs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            total_toks += len(comp.token_ids)
            if comp.finish_reason == "length":
                truncated += 1
            rep = detect_repetition_loop(txt)
            if rep["is_loop"]:
                looping += 1
            _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
            if closed:
                closed_think += 1
            pred = extract_math_prediction(txt, prefer_after_think=True, prompt_opened_think=opened)
            if math_answers_equal(pred, item["answer"]):
                correct += 1
        n = len(items)
        return {
            "n": n,
            "accuracy": round(100.0 * correct / n, 2),
            "correct": correct,
            "truncation_pct": round(100.0 * truncated / n, 2),
            "loop_pct": round(100.0 * looping / n, 2),
            "closed_think_pct": round(100.0 * closed_think / n, 2),
            "avg_tokens": round(total_toks / n, 1),
        }

    result: Dict[str, Any] = {
        "gsm8k": _eval_math_suite(gsm_items),
        "math": _eval_math_suite(math_items),
    }

    if ifeval_items:
        prompts = [_build_plain_prompt(tokenizer, it["prompt"], enable_thinking) for it in ifeval_items]
        outputs = llm.generate(prompts, sampling)
        p_pass = 0
        i_pass = 0
        i_tot = 0
        trunc = 0
        loops = 0
        toks = 0
        for item, p_str, out in zip(ifeval_items, prompts, outputs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            toks += len(comp.token_ids)
            if comp.finish_reason == "length":
                trunc += 1
            if detect_repetition_loop(txt)["is_loop"]:
                loops += 1
            g = grade_ifeval_sample(txt, item["instruction_id_list"], item["kwargs"], prompt_opened_think=opened)
            if g["prompt_strict"]:
                p_pass += 1
            i_pass += g["inst_passed"]
            i_tot += g["inst_total"]
        n = len(ifeval_items)
        result["ifeval"] = {
            "n": n,
            "prompt_strict_acc": round(100.0 * p_pass / n, 2),
            "inst_strict_acc": round(100.0 * i_pass / max(1, i_tot), 2),
            "truncation_pct": round(100.0 * trunc / n, 2),
            "loop_pct": round(100.0 * loops / n, 2),
            "avg_tokens": round(toks / n, 1),
        }

    # Clean up vLLM process/memory
    del llm
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return result


# --- STAGE 1: Triage candidate models ---

@app.function(image=image, gpu=GPU_TYPE, cpu=2, memory=16384, volumes={VOL_PATH: vol}, timeout=1800)
def triage_model(model_id: str, n_gsm8k: int = 60, n_math: int = 60, max_tokens: int = 4096) -> Dict[str, Any]:
    _guard_budget(f"triage:{model_id}")
    t0 = time.time()
    local_dir = _ensure_local_snapshot(model_id)
    ds = _load_benchmarks()
    metrics = _run_vllm_eval_on_model(
        local_dir,
        gsm_items=ds["gsm_val"][:n_gsm8k],
        math_items=ds["math_val"][:n_math],
        ifeval_items=None,
        max_tokens=max_tokens,
        enable_thinking=True,
    )
    elapsed = time.time() - t0
    cost = _record_spend(f"triage:{model_id}", GPU_TYPE, elapsed, metrics)
    out = {"model_id": model_id, "elapsed_sec": round(elapsed, 1), "cost_usd": round(cost, 4), "metrics": metrics}
    out_path = Path(f"{VOL_PATH}/triage/{_slug(model_id)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    vol.commit()
    return out


# --- STAGE 2: Mechanistic Weight-Space Diagnosis (Zero Gradient Descent!) ---

@app.function(image=image, gpu=GPU_TYPE, cpu=2, memory=16384, volumes={VOL_PATH: vol}, timeout=2400)
def diagnose_model(model_id: str, n_practice: int = 50, max_tokens: int = 2048) -> Dict[str, Any]:
    """Generate practice traces via vLLM, then extract contrastive loop vs healthy
    residual & MLP down_proj activations across ALL layers via pure forward passes."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from vllm import LLM, SamplingParams
    from eval_utils import detect_repetition_loop, extract_math_prediction, math_answers_equal, strip_thinking
    from wsp_surgery import get_language_layers

    _guard_budget(f"diagnose:{model_id}")
    t0 = time.time()
    local_dir = _ensure_local_snapshot(model_id)
    ds = _load_benchmarks()
    practice = ds["gsm_practice"][: n_practice // 2] + ds["math_practice"][: n_practice - (n_practice // 2)]

    # 1. Fast vLLM rollout on practice set ONLY (zero test leakage)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    llm = LLM(
        model=local_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
        max_model_len=max(max_tokens + 2048, 6144),
        trust_remote_code=True,
    )
    prompts = [_build_math_prompt(tokenizer, it["problem"], enable_thinking=True) for it in practice]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=max_tokens),
    )

    loop_samples: List[Tuple[str, str]] = []
    healthy_samples: List[Tuple[str, str]] = []
    for item, p_str, out in zip(practice, prompts, outputs):
        comp = out.outputs[0]
        txt = comp.text
        opened = "<think>" in p_str
        rep = detect_repetition_loop(txt)
        _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
        pred = extract_math_prediction(txt, prefer_after_think=True, prompt_opened_think=opened)
        is_ok = closed and math_answers_equal(pred, item["answer"]) and not rep["is_loop"]
        if rep["is_loop"] or comp.finish_reason == "length" or not closed:
            loop_samples.append((p_str, txt))
        elif is_ok:
            healthy_samples.append((p_str, txt))

    # If few naturally healthy samples exist (e.g. 98% loop rate on 0.8B!), collect clean non-looping
    # trajectories using the SAME model with stronger anti-repeat decoding ONLY on practice prompts
    # so we have rich contrastive healthy reasoning & </think> termination states!
    if len(healthy_samples) < 15:
        alt_outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.8,
                top_p=0.95,
                presence_penalty=2.0,
                repetition_penalty=1.1,
                max_tokens=4096,
                seed=42,
            ),
        )
        correct_healthy: List[Tuple[str, str]] = []
        other_healthy: List[Tuple[str, str]] = []
        for item, p_str, out in zip(practice, prompts, alt_outputs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            rep = detect_repetition_loop(txt)
            _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
            pred = extract_math_prediction(txt, prefer_after_think=True, prompt_opened_think=opened)
            if closed and not rep["is_loop"]:
                if math_answers_equal(pred, item["answer"]):
                    correct_healthy.append((p_str, txt))
                else:
                    other_healthy.append((p_str, txt))
        healthy_samples.extend(correct_healthy)
        if len(healthy_samples) < 15:
            healthy_samples.extend(other_healthy)

    # Final fallback: non-thinking clean completions on same practice prompts (always clean & terminating)
    if len(healthy_samples) < 10:
        nt_prompts = [_build_math_prompt(tokenizer, it["problem"], enable_thinking=False) for it in practice[:24]]
        nt_outputs = llm.generate(
            nt_prompts,
            SamplingParams(temperature=0.0, presence_penalty=1.0, max_tokens=1024),
        )
        for p_str, out in zip(nt_prompts, nt_outputs):
            txt = out.outputs[0].text
            if not detect_repetition_loop(txt)["is_loop"] and len(txt) > 30:
                healthy_samples.append((p_str, txt))

    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    loop_samples = loop_samples[:24]
    healthy_samples = healthy_samples[:24]
    if len(loop_samples) < 3 or len(healthy_samples) < 3:
        raise RuntimeError(
            f"Insufficient contrastive samples for {model_id}: loop={len(loop_samples)}, healthy={len(healthy_samples)}"
        )

    # 2. Pure forward-pass PyTorch activation probing across every layer
    cfg = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    if getattr(cfg, "model_type", "") in ("qwen3_5", "qwen3_5_moe", "gemma4", "gemma3", "qwen2_5_vl", "qwen2_vl"):
        model = AutoModelForImageTextToText.from_pretrained(
            local_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
    model.eval()
    total_params = sum(int(p.numel()) for p in model.parameters())
    layers, layer_prefix = get_language_layers(model)
    n_layers = len(layers)

    # Buffers for per-layer statistics (pure within-<think> productive vs looping contrast!)
    d_model = layers[0].mlp.down_proj.weight.shape[0]
    d_ff = layers[0].mlp.down_proj.weight.shape[1]

    loop_out_vecs: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    healthy_out_vecs: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    resid_loop_vecs: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    resid_healthy_vecs: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    term_out_vecs: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    loop_in_sum = [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)]
    loop_in_cnt = [0 for _ in range(n_layers)]
    healthy_in_sum = [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)]
    healthy_in_cnt = [0 for _ in range(n_layers)]

    # Capture BOTH MLP down_proj (z, y) AND full residual stream (layer output h)
    mode_state = {"kind": "loop", "tok_slice": slice(0, 1)}

    hooks = []
    for li in range(n_layers):
        def _make_mlp_hook(layer_idx: int):
            def _hook(_mod, inp, out):
                sl = mode_state["tok_slice"]
                z = inp[0][0, sl, :].detach().to(torch.float32).cpu()
                y = out[0, sl, :].detach().to(torch.float32).cpu()
                if z.shape[0] == 0:
                    return
                if mode_state["kind"] == "loop":
                    loop_out_vecs[layer_idx].append(y)
                    loop_in_sum[layer_idx] += z.to(torch.float64).sum(dim=0)
                    loop_in_cnt[layer_idx] += z.shape[0]
                elif mode_state["kind"] == "healthy":
                    healthy_out_vecs[layer_idx].append(y)
                    healthy_in_sum[layer_idx] += z.to(torch.float64).sum(dim=0)
                    healthy_in_cnt[layer_idx] += z.shape[0]
                elif mode_state["kind"] == "term":
                    term_out_vecs[layer_idx].append(y)
            return _hook

        def _make_resid_hook(layer_idx: int):
            def _hook(_mod, _inp, out):
                sl = mode_state["tok_slice"]
                h = (out[0] if isinstance(out, tuple) else out)[0, sl, :].detach().to(torch.float32).cpu()
                if h.shape[0] == 0:
                    return
                if mode_state["kind"] == "loop":
                    resid_loop_vecs[layer_idx].append(h)
                elif mode_state["kind"] == "healthy":
                    resid_healthy_vecs[layer_idx].append(h)
            return _hook

        hooks.append(layers[li].mlp.down_proj.register_forward_hook(_make_mlp_hook(li)))
        hooks.append(layers[li].register_forward_hook(_make_resid_hook(li)))

    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_end_tok = think_end_ids[0] if think_end_ids else None
    lm_head_w = model.get_output_embeddings().weight.detach().to(torch.float32).cpu()
    if think_end_tok is not None and think_end_tok < lm_head_w.shape[0]:
        think_end_vec = lm_head_w[think_end_tok] - lm_head_w.mean(dim=0)
        think_end_vec = think_end_vec / (think_end_vec.norm() + 1e-12)
    else:
        think_end_vec = torch.zeros(d_model, dtype=torch.float32)

    with torch.no_grad():
        # CRITICAL MECHANISTIC CONTRAST:
        # Compare productive early <think> tokens (tokens 24..180 of the SAME thinking trajectory!)
        # against degenerate late looping tokens (last 192 tokens of the SAME trajectory)!
        # Zero thinking-vs-non-thinking domain shift!
        for p_str, gen_txt in loop_samples:
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            g_ids = tokenizer.encode(gen_txt, add_special_tokens=False)[:1800]
            if len(g_ids) < 400:
                continue
            full = torch.tensor([p_ids + g_ids], device="cuda")

            # Early productive reasoning tokens (same prompt, same <think> mode, before loop begins)
            prod_start = len(p_ids) + 24
            prod_end = len(p_ids) + min(200, len(g_ids) // 4)
            mode_state["kind"] = "healthy"
            mode_state["tok_slice"] = slice(prod_start, prod_end, 3)
            model(input_ids=full, use_cache=False)

            # Late looping tokens (deep inside degenerate repetition loop)
            loop_start = len(p_ids) + len(g_ids) - 192
            mode_state["kind"] = "loop"
            mode_state["tok_slice"] = slice(loop_start, len(p_ids) + len(g_ids), 3)
            model(input_ids=full, use_cache=False)

        # Also probe genuine </think> termination states from any clean thinking completions
        for p_str, gen_txt in healthy_samples:
            if "<think>" not in p_str:
                continue
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            g_ids = tokenizer.encode(gen_txt, add_special_tokens=False)[:1800]
            if think_end_tok is not None and think_end_tok in g_ids:
                full = torch.tensor([p_ids + g_ids], device="cuda")
                t_pos = len(p_ids) + g_ids.index(think_end_tok)
                mode_state["kind"] = "term"
                mode_state["tok_slice"] = slice(max(len(p_ids), t_pos - 8), min(len(p_ids) + len(g_ids), t_pos + 4))
                model(input_ids=full, use_cache=False)

    for h in hooks:
        h.remove()

    # 3. Compute exact per-layer RESIDUAL + MLP loop directions, PCA subspaces, and culprit scores
    layer_types = getattr(getattr(cfg, "text_config", cfg), "layer_types", None) or ["full_attention"] * n_layers
    diag_layers: Dict[int, Dict[str, Any]] = {}
    layer_ranking: List[Dict[str, Any]] = []

    for li in range(n_layers):
        Y_loop = torch.cat(loop_out_vecs[li], dim=0)
        Y_healthy = torch.cat(healthy_out_vecs[li], dim=0)
        H_loop = torch.cat(resid_loop_vecs[li], dim=0)
        H_healthy = torch.cat(resid_healthy_vecs[li], dim=0)

        # MLP-output loop direction
        mu_l = Y_loop.mean(dim=0)
        mu_h = Y_healthy.mean(dim=0)
        diff = mu_l - mu_h
        diff_norm = float(diff.norm().item())
        v_dir = diff / (diff_norm + 1e-12)

        # RESIDUAL-STREAM loop direction (pure within-<think> looping minus productive reasoning!)
        h_mu_l = H_loop.mean(dim=0)
        h_mu_h = H_healthy.mean(dim=0)
        h_diff = h_mu_l - h_mu_h
        h_diff_norm = float(h_diff.norm().item())
        r_dir = h_diff / (h_diff_norm + 1e-12)

        proj_l = (H_loop @ r_dir).numpy()
        proj_h = (H_healthy @ r_dir).numpy()
        sep_d = float((proj_l.mean() - proj_h.mean()) / (np.sqrt(0.5 * (proj_l.var() + proj_h.var())) + 1e-8))

        def _top_pca(X: torch.Tensor, ref_mu: torch.Tensor) -> Tuple[torch.Tensor, float]:
            c = X - ref_mu.unsqueeze(0)
            if c.shape[0] > 512:
                c = c[:: max(1, c.shape[0] // 512)]
            _, S_p, Vh_p = torch.linalg.svd(c, full_matrices=False)
            return Vh_p[:4].contiguous(), float(S_p[0].item())

        pca_dirs, sv1 = _top_pca(Y_loop, mu_h)
        resid_pca_dirs, r_sv1 = _top_pca(H_loop, h_mu_h)

        if term_out_vecs[li]:
            Y_term = torch.cat(term_out_vecs[li], dim=0)
            t_diff = Y_term.mean(dim=0) - mu_l
            term_dir = t_diff / (t_diff.norm() + 1e-12)
        else:
            term_dir = think_end_vec.clone()

        # Exact column culprit scores on down_proj.weight [d_model, d_ff]
        W_down = layers[li].mlp.down_proj.weight.detach().to(torch.float32).cpu()
        w_align_mlp = v_dir @ W_down
        w_align_resid = r_dir @ W_down
        z_loop_mean = (loop_in_sum[li] / max(1, loop_in_cnt[li])).to(torch.float32)
        z_healthy_mean = (healthy_in_sum[li] / max(1, healthy_in_cnt[li])).to(torch.float32)
        z_diff = z_loop_mean - z_healthy_mean  # >0: neurons that fire MORE during looping than productive reasoning!
        act_weighted = z_diff * w_align_mlp
        resid_act_weighted = z_diff * w_align_resid

        top64_contrib = float(torch.topk(resid_act_weighted, k=min(64, d_ff)).values.sum().item())
        diag_layers[li] = {
            "layer_type": str(layer_types[li]) if li < len(layer_types) else "unknown",
            "loop_dir": v_dir,
            "resid_loop_dir": r_dir,
            "pca_dirs": pca_dirs,
            "resid_pca_dirs": resid_pca_dirs,
            "term_dir": term_dir,
            "think_end_dir": think_end_vec,
            "col_scores_act_weighted": act_weighted,
            "col_scores_resid_act_weighted": resid_act_weighted,
            "col_scores_z_diff": z_diff,
            "col_scores_z_excess": z_loop_mean.abs() - z_healthy_mean.abs(),
            "z_loop_mean": z_loop_mean,
            "z_healthy_mean": z_healthy_mean,
            "col_scores_weight_aligned": w_align_resid,
            "diff_norm": h_diff_norm,
            "cohen_d": sep_d,
            "top64_contrib": top64_contrib,
        }
        layer_ranking.append(
            {
                "layer": li,
                "layer_type": diag_layers[li]["layer_type"],
                "cohen_d": round(sep_d, 3),
                "resid_diff_norm": round(h_diff_norm, 4),
                "top64_contrib": round(top64_contrib, 4),
                "resid_sv1": round(r_sv1, 3),
                "max_z_diff": round(float(z_diff.max().item()), 4),
            }
        )

    layer_ranking_sorted = sorted(layer_ranking, key=lambda x: x["top64_contrib"], reverse=True)
    diag_dir = Path(f"{VOL_PATH}/diag/{_slug(model_id)}")
    diag_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "total_params": total_params,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "layer_prefix": layer_prefix,
            "n_loop_samples": len(loop_samples),
            "n_healthy_samples": len(healthy_samples),
            "layers": diag_layers,
        },
        str(diag_dir / "diagnosis.pt"),
    )

    elapsed = time.time() - t0
    cost = _record_spend(f"diagnose:{model_id}", GPU_TYPE, elapsed, {"loop": len(loop_samples), "healthy": len(healthy_samples)})
    summary = {
        "model_id": model_id,
        "total_params": total_params,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_loop_samples": len(loop_samples),
        "n_healthy_samples": len(healthy_samples),
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(cost, 4),
        "top_layers_by_contrib": layer_ranking_sorted[:10],
        "all_layers": layer_ranking,
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vol.commit()
    return summary


# --- STAGE 3: Parallel Zero-Gradient Patch Validation ---

@app.function(image=image, gpu=GPU_TYPE, cpu=2, memory=16384, volumes={VOL_PATH: vol}, timeout=1800)
def eval_patch_candidate(
    model_id: str,
    patch_spec: Dict[str, Any],
    val_gsm8k: int = 50,
    val_math: int = 50,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    import shutil
    import torch
    from wsp_surgery import apply_patch_to_checkpoint

    _guard_budget(f"search:{model_id}")
    t0 = time.time()
    base_dir = _ensure_local_snapshot(model_id)
    diag = torch.load(f"{VOL_PATH}/diag/{_slug(model_id)}/diagnosis.pt", map_location="cpu", weights_only=False)
    ds = _load_benchmarks()

    tmp_dir = f"/tmp/wsp_cand_{os.getpid()}_{int(time.time() * 1000)}"
    try:
        patch_info = apply_patch_to_checkpoint(base_dir, tmp_dir, patch_spec, diag, copy_files=False)
        metrics = _run_vllm_eval_on_model(
            tmp_dir,
            gsm_items=ds["gsm_val"][:val_gsm8k],
            math_items=ds["math_val"][:val_math],
            ifeval_items=None,
            max_tokens=max_tokens,
            enable_thinking=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    cost = _record_spend(f"search:{model_id}", GPU_TYPE, elapsed, {"name": patch_spec.get("name", "")})
    return {
        "name": patch_spec.get("name", ""),
        "patch_spec": patch_spec,
        "modified_params": patch_info["modified_params"],
        "pct_modified": patch_info["pct_modified"],
        "metrics": metrics,
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(cost, 4),
    }


# --- STAGE 4: Full Held-Out Benchmark (MATH-500, GSM8K-1319, IFEval-541) & HF Packaging ---

@app.function(image=image, gpu=GPU_TYPE, cpu=2, memory=16384, volumes={VOL_PATH: vol}, timeout=5400)
def full_eval_and_package(
    model_id: str,
    patch_spec: Dict[str, Any],
    release_slug: str,
    eval_base: bool = True,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    import torch
    from wsp_surgery import apply_patch_to_checkpoint

    _guard_budget(f"full_eval:{release_slug}")
    t0 = time.time()
    base_dir = _ensure_local_snapshot(model_id)
    diag = torch.load(f"{VOL_PATH}/diag/{_slug(model_id)}/diagnosis.pt", map_location="cpu", weights_only=False)
    ds = _load_benchmarks()

    release_dir = f"{VOL_PATH}/releases/{release_slug}"
    patch_info = apply_patch_to_checkpoint(base_dir, release_dir, patch_spec, diag, copy_files=True)

    # Save minimal standalone reproducible patch payload (few hundred KB)
    layers_compact = {
        int(li): {
            k: diag["layers"][int(li)][k]
            for k in (
                "loop_dir",
                "resid_loop_dir",
                "pca_dirs",
                "resid_pca_dirs",
                "term_dir",
                "think_end_dir",
                "col_scores_act_weighted",
                "col_scores_resid_act_weighted",
                "col_scores_z_diff",
                "col_scores_z_excess",
                "z_loop_mean",
                "z_healthy_mean",
                "col_scores_weight_aligned",
            )
            if k in diag["layers"][int(li)]
        }
        for li in patch_spec["layers"]
    }
    torch.save(
        {"model_id": model_id, "patch_spec": patch_spec, "layers": layers_compact},
        f"{release_dir}/wsp_patch.pt",
    )

    results: Dict[str, Any] = {
        "model_id": model_id,
        "release_slug": release_slug,
        "patch_info": patch_info,
        "max_tokens": max_tokens,
    }

    if eval_base:
        results["base"] = _run_vllm_eval_on_model(
            base_dir,
            gsm_items=ds["gsm_test"],
            math_items=ds["math_test"],
            ifeval_items=ds["ifeval_test"],
            max_tokens=max_tokens,
            enable_thinking=True,
        )

    results["patched"] = _run_vllm_eval_on_model(
        release_dir,
        gsm_items=ds["gsm_test"],
        math_items=ds["math_test"],
        ifeval_items=ds["ifeval_test"],
        max_tokens=max_tokens,
        enable_thinking=True,
    )

    elapsed = time.time() - t0
    cost = _record_spend(f"full_eval:{release_slug}", GPU_TYPE, elapsed, {"release": release_slug})
    results["elapsed_sec"] = round(elapsed, 1)
    results["cost_usd"] = round(cost, 4)

    Path(f"{release_dir}/eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    vol.commit()
    return results


@app.function(image=image, gpu="H100", cpu=2, memory=16384, volumes={VOL_PATH: vol}, timeout=5400)
def eval_side_h100(model_path: str, tag: str, max_tokens: int = 4096) -> Dict[str, Any]:
    _guard_budget(f"full_h100:{tag}")
    t0 = time.time()
    vol.reload()
    ds = _load_benchmarks()
    metrics = _run_vllm_eval_on_model(
        model_path,
        gsm_items=ds["gsm_test"],
        math_items=ds["math_test"],
        ifeval_items=ds["ifeval_test"],
        max_tokens=max_tokens,
        enable_thinking=True,
    )
    elapsed = time.time() - t0
    cost = _record_spend(f"full_h100:{tag}", "H100", elapsed, {"tag": tag})
    return {"tag": tag, "metrics": metrics, "elapsed_sec": round(elapsed, 1), "cost_usd": round(cost, 4)}


@app.function(image=image, cpu=2, memory=8192, volumes={VOL_PATH: vol}, timeout=900)
def package_release_only(model_id: str, patch_spec: Dict[str, Any], release_slug: str) -> Dict[str, Any]:
    import torch
    from wsp_surgery import apply_patch_to_checkpoint

    vol.reload()
    base_dir = _ensure_local_snapshot(model_id)
    diag = torch.load(f"{VOL_PATH}/diag/{_slug(model_id)}/diagnosis.pt", map_location="cpu", weights_only=False)
    release_dir = f"{VOL_PATH}/releases/{release_slug}"
    patch_info = apply_patch_to_checkpoint(base_dir, release_dir, patch_spec, diag, copy_files=True)
    layers_compact = {
        int(li): {
            k: diag["layers"][int(li)][k]
            for k in (
                "loop_dir",
                "resid_loop_dir",
                "pca_dirs",
                "resid_pca_dirs",
                "term_dir",
                "think_end_dir",
                "col_scores_act_weighted",
                "col_scores_resid_act_weighted",
                "col_scores_z_diff",
                "col_scores_z_excess",
                "z_loop_mean",
                "z_healthy_mean",
                "col_scores_weight_aligned",
            )
            if k in diag["layers"][int(li)]
        }
        for li in patch_spec["layers"]
    }
    torch.save(
        {"model_id": model_id, "patch_spec": patch_spec, "layers": layers_compact},
        f"{release_dir}/wsp_patch.pt",
    )
    vol.commit()
    return {"base_dir": base_dir, "release_dir": release_dir, "patch_info": patch_info}


@app.function(image=image, cpu=1, memory=1024, volumes={VOL_PATH: vol}, timeout=120)
def read_budget() -> Dict[str, Any]:
    vol.reload()
    p = Path(SPEND_LOG)
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    total = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    return {
        "total_spent_usd": round(total, 4),
        "remaining_under_45_cap": round(max(0.0, HARD_BUDGET_CAP_USD - total), 4),
        "remaining_under_50_budget": round(max(0.0, 50.0 - total), 4),
        "num_runs": len(rows),
        "runs": rows[-20:],
    }


@app.local_entrypoint()
def main(
    step: str = "budget",
    models: str = "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B",
    model: str = "Qwen/Qwen3.5-0.8B",
    specs_file: str = "",
    release_slug: str = "",
    eval_base: bool = True,
    max_tokens: int = 4096,
):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    if step == "budget":
        info = read_budget.remote()
        print(json.dumps(info, indent=2))
        return

    if step == "triage":
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        res = list(triage_model.map(model_list))
        (out_dir / "triage.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if step == "diagnose":
        model_list = [m.strip() for m in (models if "," in models else model).split(",") if m.strip()]
        if len(model_list) == 1:
            res = [diagnose_model.remote(model_list[0])]
        else:
            res = list(diagnose_model.map(model_list))
        for item in res:
            (out_dir / f"diag_{_slug(item['model_id'])}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if step == "search":
        assert specs_file, "--specs-file is required for step=search"
        specs = json.loads(Path(specs_file).read_text(encoding="utf-8"))
        res = list(
            eval_patch_candidate.starmap([(model, sp, 50, 50, max_tokens) for sp in specs])
        )
        res.sort(
            key=lambda r: (
                r["metrics"]["math"]["accuracy"] + r["metrics"]["gsm8k"]["accuracy"],
                -r["metrics"]["math"]["loop_pct"],
            ),
            reverse=True,
        )
        out_path = out_dir / f"search_{_slug(model)}_{Path(specs_file).stem}.json"
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if step == "full":
        assert specs_file and release_slug, "--specs-file and --release-slug required for step=full"
        spec = json.loads(Path(specs_file).read_text(encoding="utf-8"))
        res = full_eval_and_package.remote(model, spec, release_slug, eval_base=eval_base, max_tokens=max_tokens)
        (out_dir / f"full_{release_slug}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if step == "full_parallel":
        assert specs_file and release_slug, "--specs-file and --release-slug required for step=full_parallel"
        spec = json.loads(Path(specs_file).read_text(encoding="utf-8"))
        pkg = package_release_only.remote(model, spec, release_slug)
        sides = list(
            eval_side_h100.starmap(
                [
                    (pkg["base_dir"], f"{release_slug}:base", max_tokens),
                    (pkg["release_dir"], f"{release_slug}:patched", max_tokens),
                ]
            )
        )
        res = {
            "model_id": model,
            "release_slug": release_slug,
            "patch_info": pkg["patch_info"],
            "max_tokens": max_tokens,
            "base": sides[0]["metrics"],
            "patched": sides[1]["metrics"],
            "elapsed_sec": round(sides[0]["elapsed_sec"] + sides[1]["elapsed_sec"], 1),
            "cost_usd": round(sides[0]["cost_usd"] + sides[1]["cost_usd"], 4),
        }
        (out_dir / f"full_{release_slug}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    raise ValueError(f"Unknown step: {step}")
