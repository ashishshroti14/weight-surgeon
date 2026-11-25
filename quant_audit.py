"""Quantization Integrity Audit for Qwen3.5 Base vs. Zero-Gradient WSP Models.

Tests whether common post-training quantizations (FP8, BitsAndBytes NF4 4-bit,
W4A16-RTN, and calibration-based W4A16-AWQ) preserve or erase the surgical
0.0056%–0.0118% zero-gradient Weight-Space Programming (WSP) edits in:
  - Qwen/Qwen3.5-2B  vs.  ashishshroti14/Qwen3.5-2B-WSP  (layers [15, 19], 64 down_proj cols)
  - Qwen/Qwen3.5-9B  vs.  ashishshroti14/Qwen3.5-9B-WSP  (layers [23, 27], 64 down_proj cols)

Hard rules enforced by this script:
  1. Ephemeral storage ONLY — zero Modal Volumes (`modal.Volume` is never used).
  2. Strict hard budget cap: `HARD_BUDGET_CAP_USD = 12.0` (software pre-flight +
     per-wave cumulative spend guard + hard per-container timeouts).
  3. Runs on Modal GPUs:
       modal run quant_audit.py --sizes 2B,9B --quants bf16,fp8,bnb_4bit,w4a16_rtn
  4. Zero-cost ($0 GPU) local CPU shard-level quantization SNR audit included:
       python quant_audit.py --cpu-snr-only --sizes 2B,9B
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import modal
except ImportError:  # Allows --cpu-snr-only locally even without modal installed
    modal = None  # type: ignore[assignment]

APP_NAME = "wsp-quant-audit"
HARD_BUDGET_CAP_USD = 12.0

# All-in Modal Starter hourly rates for gpu + cpu=4 + memory=24576
# (2-CPU/16GiB base rates $2.17/$4.17 + ~$0.15/hr for +2 CPU & +8 GiB RAM)
GPU_RATES_PER_HR: Dict[str, float] = {
    "L40S": 2.32,
    "H100": 4.32,
}

# Hard per-container timeout caps (seconds)
TIMEOUT_SEC_BY_GPU: Dict[str, int] = {
    "L40S": 1800,
    "H100": 1500,
}

SUPPORTED_QUANTS = ("bf16", "fp8", "bnb_4bit", "w4a16_rtn", "w4a16_awq")

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "0.8B": {
        "base": "Qwen/Qwen3.5-0.8B",
        "wsp": "ashishshroti14/Qwen3.5-0.8B-WSP",
        "edited_layers": [15, 19],
        "default_gpu": "L40S",
        "est_min_per_cell": {
            "bf16": 3.5,
            "fp8": 3.5,
            "bnb_4bit": 4.0,
            "w4a16_rtn": 4.0,
            "w4a16_awq": 5.5,
        },
    },
    "2B": {
        "base": "Qwen/Qwen3.5-2B",
        "wsp": "ashishshroti14/Qwen3.5-2B-WSP",
        "edited_layers": [15, 19],
        "default_gpu": "L40S",
        "est_min_per_cell": {
            "bf16": 4.5,
            "fp8": 4.5,
            "bnb_4bit": 5.5,
            "w4a16_rtn": 5.5,
            "w4a16_awq": 7.5,
        },
    },
    "4B": {
        "base": "Qwen/Qwen3.5-4B",
        "wsp": "ashishshroti14/Qwen3.5-4B-WSP",
        "edited_layers": [23, 26, 27],
        "default_gpu": "L40S",
        "est_min_per_cell": {
            "bf16": 5.0,
            "fp8": 5.0,
            "bnb_4bit": 6.0,
            "w4a16_rtn": 6.0,
            "w4a16_awq": 8.0,
        },
    },
    "9B": {
        "base": "Qwen/Qwen3.5-9B",
        "wsp": "ashishshroti14/Qwen3.5-9B-WSP",
        "edited_layers": [23, 27],
        "default_gpu": "H100",
        "est_min_per_cell": {
            "bf16": 5.0,
            "fp8": 5.0,
            "bnb_4bit": 6.5,
            "w4a16_rtn": 6.5,
            "w4a16_awq": 9.5,
        },
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_UTILS_PATH = SCRIPT_DIR / "eval_utils.py"
RESULTS_DIR = SCRIPT_DIR / "results"
SPEND_LEDGER_PATH = RESULTS_DIR / "quant_audit_spend.jsonl"

# Exact 16-level bitsandbytes NormalFloat4 (NF4) codebook
NF4_CODEBOOK: Tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


# ---------------------------------------------------------------------------
# Ephemeral Modal App & Image (NO persistent volumes)
# ---------------------------------------------------------------------------

if modal is not None:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.34.0",
            "transformers",
            "datasets",
            "safetensors",
            "huggingface_hub",
            "hf_transfer",
            "sympy",
            "antlr4-python3-runtime==4.11.1",
            "langdetect",
            "einops",
            "pillow",
            "qwen-vl-utils",
            "loguru",
            "nvidia-ml-py",
        )
        .run_commands(
            "python -m pip install --no-deps accelerate 'bitsandbytes>=0.46.1' vllm-bnb-plugin 'compressed-tensors==0.20.1' llmcompressor",
            "python -c \"import vllm, bitsandbytes, accelerate, llmcompressor; from vllm import LLM; from llmcompressor import oneshot; from llmcompressor.modifiers.quantization import QuantizationModifier; print('IMAGE_IMPORTS_OK', vllm.__version__)\"",
        )
        .env(
            {
                "HF_HOME": "/tmp/hf_cache",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        )
        .add_local_file(str(EVAL_UTILS_PATH), "/root/eval_utils.py")
    )
else:
    app = None
    image = None


# ---------------------------------------------------------------------------
# Shared Prompting & Dataset Helpers (exact parity with wsp_modal.py)
# ---------------------------------------------------------------------------

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


def _load_audit_subsets(
    n_gsm8k: int = 300,
    n_math: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Load deterministic subsets of GSM8K test and MATH-500 test."""
    from datasets import load_dataset
    from eval_utils import extract_gsm8k_gold

    # GSM8K test (n=1319 total) -> deterministic seeded subset
    gsm_raw = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng_gsm = random.Random(seed)
    gsm_idxs = list(range(len(gsm_raw)))
    rng_gsm.shuffle(gsm_idxs)
    gsm_sel = sorted(gsm_idxs[: min(n_gsm8k, len(gsm_raw))])
    gsm_items = [
        {
            "problem": gsm_raw[i]["question"],
            "answer": extract_gsm8k_gold(gsm_raw[i]["answer"]),
        }
        for i in gsm_sel
    ]

    # MATH-500 test (n=500 total) -> stratified by difficulty level if available
    math_raw = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if n_math >= len(math_raw):
        math_sel = list(range(len(math_raw)))
    elif math_raw and "level" in math_raw[0]:
        by_lvl: Dict[Any, List[int]] = {}
        for idx, row in enumerate(math_raw):
            by_lvl.setdefault(row["level"], []).append(idx)
        levels = sorted(by_lvl.keys())
        base_per = n_math // len(levels)
        rem = n_math % len(levels)
        math_sel = []
        for l_i, lvl in enumerate(levels):
            pool = list(by_lvl[lvl])
            random.Random(seed + 100 + l_i).shuffle(pool)
            take = base_per + (1 if l_i < rem else 0)
            math_sel.extend(pool[: min(take, len(pool))])
        # Fill any shortfall deterministically if a level had fewer rows
        if len(math_sel) < n_math:
            chosen = set(math_sel)
            rest = [i for i in range(len(math_raw)) if i not in chosen]
            random.Random(seed + 999).shuffle(rest)
            math_sel.extend(rest[: n_math - len(math_sel)])
        math_sel.sort()
    else:
        math_idxs = list(range(len(math_raw)))
        random.Random(seed + 100).shuffle(math_idxs)
        math_sel = sorted(math_idxs[:n_math])

    math_items = [
        {"problem": math_raw[i]["problem"], "answer": math_raw[i]["answer"]}
        for i in math_sel
    ]
    return {"gsm8k": gsm_items, "math500": math_items}


# ---------------------------------------------------------------------------
# Ephemeral Offline PTQ (llmcompressor W4A16-RTN / W4A16-AWQ) into /tmp
# ---------------------------------------------------------------------------

def _quantize_w4a16_to_tmp(model_id: str, quant_mode: str, num_cal_samples: int = 128) -> str:
    """Create an ephemeral W4A16 compressed-tensors checkpoint in /tmp for vLLM Marlin."""
    import torch
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    out_dir = tempfile.mkdtemp(prefix=f"wsp_{quant_mode}_", dir="/tmp")
    ignore_modules = [
        "re:.*lm_head",
        "re:.*visual.*",
        r"re:.*embed_tokens$",
        r"re:.*linear_attn\.in_proj_a$",
        r"re:.*linear_attn\.in_proj_b$",
    ]

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        processor = None

    if quant_mode == "w4a16_rtn":
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=ignore_modules,
        )
        oneshot(model=model, recipe=recipe)
    elif quant_mode == "w4a16_awq":
        # llmcompressor >= 0.8 splits transform.awq.AWQModifier (scaling only) from QuantizationModifier;
        # legacy llmcompressor.modifiers.awq.AWQModifier accepts scheme="W4A16" directly.
        try:
            from llmcompressor.modifiers.transform.awq import AWQModifier as TransformAWQModifier

            recipe = [
                TransformAWQModifier(ignore=ignore_modules),
                QuantizationModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=ignore_modules,
                ),
            ]
        except ImportError:
            from llmcompressor.modifiers.awq import AWQModifier as LegacyAWQModifier

            recipe = LegacyAWQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=ignore_modules,
            )

        # Generic non-looping WikiText calibration explicitly tests dormant-neuron blindness
        raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cal_ds = raw_ds.filter(lambda r: len(r["text"].strip()) >= 120).select(
            range(num_cal_samples)
        )
        oneshot_kwargs: Dict[str, Any] = {
            "model": model,
            "dataset": cal_ds,
            "recipe": recipe,
            "max_seq_length": 1024,
            "num_calibration_samples": num_cal_samples,
        }
        if processor is not None:
            oneshot_kwargs["processor"] = processor
        else:
            oneshot_kwargs["tokenizer"] = tokenizer
        oneshot(**oneshot_kwargs)
    else:
        raise ValueError(f"Unsupported offline W4A16 mode: {quant_mode}")

    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    if processor is not None:
        try:
            processor.save_pretrained(out_dir)
        except Exception:
            pass

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir


# ---------------------------------------------------------------------------
# Core GPU Evaluation Cell (executed on ephemeral L40S or H100)
# ---------------------------------------------------------------------------

def _run_quant_eval_cell(
    model_id: str,
    size_tag: str,
    side: str,
    quant: str,
    gpu_type: str,
    n_gsm8k: int = 300,
    n_math: int = 200,
    max_tokens: int = 4096,
    seed: int = 42,
) -> Dict[str, Any]:
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from eval_utils import (
        detect_repetition_loop,
        extract_math_prediction,
        math_answers_equal,
        strip_thinking,
    )

    t0 = time.time()
    tmp_quant_dir: Optional[str] = None
    llm = None
    try:
        subsets = _load_audit_subsets(n_gsm8k=n_gsm8k, n_math=n_math, seed=seed)
        load_path = model_id
        llm_kwargs: Dict[str, Any] = {
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.82,
            "max_model_len": max(max_tokens + 2048, 8192),
            "trust_remote_code": True,
            "enforce_eager": False,
        }

        if quant == "bf16":
            pass
        elif quant == "fp8":
            llm_kwargs["quantization"] = "fp8"
        elif quant == "bnb_4bit":
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
        elif quant in ("w4a16_rtn", "w4a16_awq"):
            tmp_quant_dir = _quantize_w4a16_to_tmp(model_id, quant_mode=quant)
            load_path = tmp_quant_dir
        else:
            raise ValueError(f"Unsupported quant mode: {quant}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        llm = LLM(model=load_path, **llm_kwargs)

        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_tokens=max_tokens,
        )

        def _eval_math(items: List[Dict[str, str]]) -> Dict[str, Any]:
            if not items:
                return {}
            prompts = [_build_math_prompt(tokenizer, it["problem"], enable_thinking=True) for it in items]
            outputs = llm.generate(prompts, sampling)
            correct = truncated = looping = closed_think = total_toks = 0
            for item, p_str, out in zip(items, prompts, outputs):
                comp = out.outputs[0]
                txt = comp.text
                opened = "<think>" in p_str
                total_toks += len(comp.token_ids)
                if comp.finish_reason == "length":
                    truncated += 1
                if detect_repetition_loop(txt)["is_loop"]:
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

        metrics = {
            "gsm8k": _eval_math(subsets["gsm8k"]),
            "math500": _eval_math(subsets["math500"]),
        }
        status = "ok"
        error_msg = None
    except Exception as exc:
        metrics = {}
        status = "error"
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            del llm
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if tmp_quant_dir and os.path.isdir(tmp_quant_dir):
            shutil.rmtree(tmp_quant_dir, ignore_errors=True)

    elapsed = time.time() - t0
    rate = GPU_RATES_PER_HR.get(gpu_type, 4.17)
    est_cost = (elapsed / 3600.0) * rate
    return {
        "status": status,
        "error": error_msg,
        "size": size_tag,
        "side": side,
        "quant": quant,
        "model_id": model_id,
        "gpu": gpu_type,
        "max_tokens": max_tokens,
        "elapsed_sec": round(elapsed, 1),
        "est_cost_usd": round(est_cost, 4),
        "metrics": metrics,
    }


if modal is not None:
    @app.function(image=image, gpu="L40S", cpu=4, memory=24576, timeout=TIMEOUT_SEC_BY_GPU["L40S"])
    def eval_quant_cell_l40s(
        model_id: str,
        size_tag: str,
        side: str,
        quant: str,
        n_gsm8k: int = 300,
        n_math: int = 200,
        max_tokens: int = 4096,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return _run_quant_eval_cell(
            model_id=model_id,
            size_tag=size_tag,
            side=side,
            quant=quant,
            gpu_type="L40S",
            n_gsm8k=n_gsm8k,
            n_math=n_math,
            max_tokens=max_tokens,
            seed=seed,
        )

    @app.function(image=image, gpu="H100", cpu=4, memory=24576, timeout=TIMEOUT_SEC_BY_GPU["H100"])
    def eval_quant_cell_h100(
        model_id: str,
        size_tag: str,
        side: str,
        quant: str,
        n_gsm8k: int = 300,
        n_math: int = 200,
        max_tokens: int = 4096,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return _run_quant_eval_cell(
            model_id=model_id,
            size_tag=size_tag,
            side=side,
            quant=quant,
            gpu_type="H100",
            n_gsm8k=n_gsm8k,
            n_math=n_math,
            max_tokens=max_tokens,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Local Budget Ledger & Guard (strictly local disk, zero Modal Volumes)
# ---------------------------------------------------------------------------

def _read_local_spend() -> float:
    if not SPEND_LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    for ln in SPEND_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            total += float(json.loads(ln).get("est_cost_usd", 0.0))
        except Exception:
            pass
    return total


def _append_local_spend(record: Dict[str, Any]) -> None:
    SPEND_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "size": record.get("size"),
        "side": record.get("side"),
        "quant": record.get("quant"),
        "gpu": record.get("gpu"),
        "status": record.get("status"),
        "elapsed_sec": record.get("elapsed_sec", 0.0),
        "est_cost_usd": record.get("est_cost_usd", 0.0),
    }
    with open(SPEND_LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def estimate_plan_cost(
    sizes: List[str],
    quants: List[str],
    gpu_override_9b: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Return (total_est_usd, wave_plan) for Base+WSP pairs across sizes and quants."""
    waves: List[Dict[str, Any]] = []
    total_usd = 0.0
    for sz in sizes:
        cfg = MODEL_REGISTRY[sz]
        gpu = gpu_override_9b if (sz == "9B" and gpu_override_9b) else cfg["default_gpu"]
        rate = GPU_RATES_PER_HR[gpu]
        for q in quants:
            est_min_cell = cfg["est_min_per_cell"][q]
            pair_usd = 2.0 * (est_min_cell / 60.0) * rate
            waves.append(
                {
                    "size": sz,
                    "quant": q,
                    "gpu": gpu,
                    "base_model": cfg["base"],
                    "wsp_model": cfg["wsp"],
                    "est_min_per_cell": est_min_cell,
                    "est_pair_usd": round(pair_usd, 4),
                }
            )
            total_usd += pair_usd
    return round(total_usd, 4), waves


# ---------------------------------------------------------------------------
# $0 CPU-Only Shard Quantization Error vs. WSP Delta Audit
# ---------------------------------------------------------------------------

def _load_down_proj_from_hf(repo_id: str, layer_indices: List[int]) -> Dict[int, Any]:
    """Download only the 1–2 safetensors shards containing target down_proj weights."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    index_file = None
    try:
        index_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    except Exception:
        index_file = None

    if index_file is not None:
        weight_map: Dict[str, str] = json.loads(Path(index_file).read_text(encoding="utf-8"))["weight_map"]
        all_keys = list(weight_map.keys())
    else:
        single_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        tensors = load_file(single_path)
        all_keys = list(tensors.keys())
        out: Dict[int, torch.Tensor] = {}
        for l_idx in layer_indices:
            suffix = f"layers.{l_idx}.mlp.down_proj.weight"
            matches = [k for k in all_keys if k.endswith(suffix) and "visual" not in k]
            matches.sort(key=lambda k: (0 if "language_model" in k else 1, len(k)))
            out[l_idx] = tensors[matches[0]].to(torch.float32)
        return out

    shard_to_layers: Dict[str, List[Tuple[int, str]]] = {}
    for l_idx in layer_indices:
        suffix = f"layers.{l_idx}.mlp.down_proj.weight"
        matches = [k for k in all_keys if k.endswith(suffix) and "visual" not in k]
        if not matches:
            raise KeyError(f"{repo_id}: could not find key ending with {suffix}")
        matches.sort(key=lambda k: (0 if "language_model" in k else 1, len(k)))
        key = matches[0]
        shard_to_layers.setdefault(weight_map[key], []).append((l_idx, key))

    out = {}
    for shard_name, items in shard_to_layers.items():
        shard_path = hf_hub_download(repo_id=repo_id, filename=shard_name)
        tensors = load_file(shard_path)
        for l_idx, key in items:
            out[l_idx] = tensors[key].to(torch.float32)
    return out


def _sim_quant_dequant(W: Any, scheme: str) -> Any:
    """Simulate realistic row-group quantization-dequantization on W [d_model, d_ff]."""
    import torch

    d_model, d_ff = W.shape
    if scheme == "fp8_e4m3_row":
        amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = amax / 448.0
        scaled = (W / scale).clamp(-448.0, 448.0)
        if hasattr(torch, "float8_e4m3fn"):
            q = scaled.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            # 3-bit mantissa fallback
            exp = torch.floor(torch.log2(scaled.abs().clamp(min=2.0**-9)))
            step = torch.pow(2.0, exp - 3.0)
            q = torch.round(scaled / step) * step
        return q * scale

    if scheme == "q8_0_b32":
        g = 32
        Wg = W.view(d_model, d_ff // g, g)
        scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
        q = torch.round(Wg / scale).clamp(-127.0, 127.0)
        return (q * scale).view(d_model, d_ff)

    if scheme == "w4a16_g128":
        g = 128
        Wg = W.view(d_model, d_ff // g, g)
        scale = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
        q = torch.round(Wg / scale).clamp(-8.0, 7.0)
        return (q * scale).view(d_model, d_ff)

    if scheme == "nf4_g64":
        g = 64
        cb = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=W.device)
        Wg = W.view(d_model, d_ff // g, g)
        absmax = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        normed = (Wg / absmax).clamp(-1.0, 1.0)
        idx = (normed.unsqueeze(-1) - cb).abs().argmin(dim=-1)
        return (cb[idx] * absmax).view(d_model, d_ff)

    if scheme == "q4_km_b32":
        # Asymmetric 4-bit block-32 along rows (approximating GGUF Q4_K_M sub-blocks)
        g = 32
        Wg = W.view(d_model, d_ff // g, g)
        wmin = Wg.amin(dim=-1, keepdim=True)
        wmax = Wg.amax(dim=-1, keepdim=True)
        scale = (wmax - wmin).clamp(min=1e-12) / 15.0
        q = torch.round((Wg - wmin) / scale).clamp(0.0, 15.0)
        return (q * scale + wmin).view(d_model, d_ff)

    raise ValueError(f"Unknown CPU quantization simulation scheme: {scheme}")


def run_cpu_snr_audit(sizes: List[str]) -> Dict[str, Any]:
    """$0 GPU local CPU audit comparing WSP edit delta vs. per-column quantization noise."""
    import torch

    schemes = ("fp8_e4m3_row", "q8_0_b32", "w4a16_g128", "nf4_g64", "q4_km_b32")
    group_sizes = {
        "fp8_e4m3_row": None,
        "q8_0_b32": 32,
        "w4a16_g128": 128,
        "nf4_g64": 64,
        "q4_km_b32": 32,
    }
    report: Dict[str, Any] = {"mode": "cpu_snr_audit", "gpu_cost_usd": 0.0, "models": {}}

    for sz in sizes:
        cfg = MODEL_REGISTRY[sz]
        layers = cfg["edited_layers"]
        base_w = _load_down_proj_from_hf(cfg["base"], layers)
        wsp_w = _load_down_proj_from_hf(cfg["wsp"], layers)
        model_rep: Dict[str, Any] = {"base": cfg["base"], "wsp": cfg["wsp"], "layers": {}}

        for l_idx in layers:
            Wb = base_w[l_idx]
            Wp = wsp_w[l_idx]
            delta = Wp - Wb
            col_diff = delta.abs().sum(dim=0)
            edited_cols = torch.nonzero(col_diff > 1e-7, as_tuple=False).squeeze(-1)
            d_model, d_ff = Wb.shape

            delta_ed = delta[:, edited_cols]
            base_ed = Wb[:, edited_cols]
            delta_l2 = float(delta_ed.norm().item())
            base_ed_l2 = float(base_ed.norm().item())

            layer_rep: Dict[str, Any] = {
                "shape": [int(d_model), int(d_ff)],
                "num_edited_cols": int(edited_cols.numel()),
                "edit_delta_l2": round(delta_l2, 6),
                "edit_to_base_col_ratio": round(delta_l2 / (base_ed_l2 + 1e-12), 4),
                "schemes": {},
            }

            edited_mask_cols = torch.zeros(d_ff, dtype=torch.bool)
            edited_mask_cols[edited_cols] = True

            for sc in schemes:
                Qb = _sim_quant_dequant(Wb, sc)
                Qp = _sim_quant_dequant(Wp, sc)
                q_err_ed = Qp[:, edited_cols] - Wp[:, edited_cols]
                q_delta_ed = Qp[:, edited_cols] - Qb[:, edited_cols]

                q_err_l2 = float(q_err_ed.norm().item())
                q_delta_l2 = float(q_delta_ed.norm().item())
                cos_sim = float(
                    torch.dot(q_delta_ed.reshape(-1), delta_ed.reshape(-1)).item()
                    / (q_delta_l2 * delta_l2 + 1e-12)
                )
                bin_collision = float((q_delta_ed.abs() < 1e-9).float().mean().item() * 100.0)

                # Collateral distortion on UNEDITED weights sharing a row-group with an edited col
                g = group_sizes[sc]
                neighbor_mse_ratio = 1.0
                if g is not None:
                    n_groups = d_ff // g
                    grp_has_edit = edited_mask_cols.view(n_groups, g).any(dim=1)
                    col_in_dirty_grp = grp_has_edit.unsqueeze(1).expand(n_groups, g).reshape(d_ff)
                    dirty_unedited = col_in_dirty_grp & (~edited_mask_cols)
                    clean_unedited = ~col_in_dirty_grp
                    mse_dirty = float(((Qp[:, dirty_unedited] - Wp[:, dirty_unedited]) ** 2).mean().item())
                    mse_clean = float(((Qp[:, clean_unedited] - Wp[:, clean_unedited]) ** 2).mean().item())
                    neighbor_mse_ratio = mse_dirty / (mse_clean + 1e-18)

                layer_rep["schemes"][sc] = {
                    "quant_err_l2_edited": round(q_err_l2, 6),
                    "snr_edit_over_qerr": round(delta_l2 / (q_err_l2 + 1e-12), 3),
                    "delta_cosine": round(cos_sim, 5),
                    "delta_norm_retention": round(q_delta_l2 / (delta_l2 + 1e-12), 4),
                    "bin_collision_pct": round(bin_collision, 2),
                    "neighbor_mse_ratio": round(neighbor_mse_ratio, 4),
                }
            model_rep["layers"][str(l_idx)] = layer_rep
        report["models"][sz] = model_rep

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "quant_cpu_snr.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved $0 CPU SNR audit -> {out_path}")
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Retention Summary Helper
# ---------------------------------------------------------------------------

def _summarize_retention(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Base vs. WSP deltas and lift retention % relative to BF16."""
    by_key = {(c["size"], c["quant"], c["side"]): c for c in cells if c.get("status") == "ok"}
    summary: Dict[str, Any] = {}
    sizes = sorted({c["size"] for c in cells})
    quants = list(dict.fromkeys(c["quant"] for c in cells))

    for sz in sizes:
        sz_sum: Dict[str, Any] = {}
        bf16_lifts: Dict[str, float] = {}
        b_ref = by_key.get((sz, "bf16", "base"))
        w_ref = by_key.get((sz, "bf16", "wsp"))
        if b_ref and w_ref:
            for bench in ("gsm8k", "math500"):
                bm = b_ref["metrics"].get(bench, {})
                wm = w_ref["metrics"].get(bench, {})
                if bm and wm:
                    bf16_lifts[bench] = round(wm["accuracy"] - bm["accuracy"], 2)

        for q in quants:
            b = by_key.get((sz, q, "base"))
            w = by_key.get((sz, q, "wsp"))
            if not b or not w:
                continue
            q_entry: Dict[str, Any] = {}
            for bench in ("gsm8k", "math500"):
                bm = b["metrics"].get(bench, {})
                wm = w["metrics"].get(bench, {})
                if not bm or not wm:
                    continue
                acc_lift = round(wm["accuracy"] - bm["accuracy"], 2)
                trunc_delta = round(wm["truncation_pct"] - bm["truncation_pct"], 2)
                closed_delta = round(wm["closed_think_pct"] - bm["closed_think_pct"], 2)
                ret_pct = None
                if bench in bf16_lifts and abs(bf16_lifts[bench]) > 1e-6:
                    ret_pct = round(100.0 * acc_lift / bf16_lifts[bench], 1)
                q_entry[bench] = {
                    "base_acc": bm["accuracy"],
                    "wsp_acc": wm["accuracy"],
                    "acc_lift_pts": acc_lift,
                    "lift_retention_vs_bf16_pct": ret_pct,
                    "base_trunc_pct": bm["truncation_pct"],
                    "wsp_trunc_pct": wm["truncation_pct"],
                    "trunc_delta_pts": trunc_delta,
                    "base_closed_think_pct": bm["closed_think_pct"],
                    "wsp_closed_think_pct": wm["closed_think_pct"],
                    "closed_think_delta_pts": closed_delta,
                    "base_avg_tokens": bm["avg_tokens"],
                    "wsp_avg_tokens": wm["avg_tokens"],
                }
            sz_sum[q] = q_entry
        summary[sz] = sz_sum
    return summary


# ---------------------------------------------------------------------------
# Modal Local Entrypoint (budget-guarded wave dispatcher)
# ---------------------------------------------------------------------------

if modal is not None:
    @app.local_entrypoint()
    def main(
        sizes: str = "2B,9B",
        quants: str = "bf16,fp8,bnb_4bit,w4a16_rtn",
        n_gsm8k: int = 300,
        n_math: int = 200,
        max_tokens: int = 4096,
        seed: int = 42,
        max_budget_usd: float = HARD_BUDGET_CAP_USD,
        gpu_9b: str = "H100",
        dry_run: bool = False,
        ignore_prior_spend: bool = False,
        out_name: str = "quant_audit_results.json",
    ) -> None:
        if max_budget_usd <= 0.0 or max_budget_usd > HARD_BUDGET_CAP_USD:
            raise ValueError(
                f"max_budget_usd={max_budget_usd} must be in (0, {HARD_BUDGET_CAP_USD}]"
            )
        if gpu_9b not in GPU_RATES_PER_HR:
            raise ValueError(f"gpu_9b must be one of {list(GPU_RATES_PER_HR)}, got {gpu_9b}")

        size_list = [s.strip().upper() for s in sizes.split(",") if s.strip()]
        quant_list = [q.strip().lower() for q in quants.split(",") if q.strip()]
        for sz in size_list:
            if sz not in MODEL_REGISTRY:
                raise ValueError(f"Unknown size '{sz}'. Choose from {list(MODEL_REGISTRY)}")
        for q in quant_list:
            if q not in SUPPORTED_QUANTS:
                raise ValueError(f"Unknown quant '{q}'. Choose from {SUPPORTED_QUANTS}")

        est_total, waves = estimate_plan_cost(size_list, quant_list, gpu_override_9b=gpu_9b)
        prior_spent = 0.0 if ignore_prior_spend else _read_local_spend()

        active_profile = os.environ.get("MODAL_PROFILE", "default")
        print(f"=== Quantization Integrity Audit Plan (profile={active_profile}, ephemeral-only) ===")
        print(f"Sizes: {size_list} | Quants: {quant_list} | Subset: GSM8K {n_gsm8k} + MATH-500 {n_math} (greedy {max_tokens} tok)")
        print(f"Prior local ledger spend: ${prior_spent:.4f} | Planned est: ${est_total:.4f} | Hard cap: ${max_budget_usd:.2f}")
        for w in waves:
            print(f"  Wave {w['size']} {w['quant']:10s} [{w['gpu']}] est ${w['est_pair_usd']:.3f} (2x {w['est_min_per_cell']}m)")

        if dry_run:
            print("[DRY RUN] No Modal GPU containers launched.")
            return

        if prior_spent + est_total > max_budget_usd:
            raise RuntimeError(
                f"BUDGET GUARD ABORT (pre-flight): prior (${prior_spent:.2f}) + est (${est_total:.2f}) "
                f"= ${prior_spent + est_total:.2f} > cap ${max_budget_usd:.2f}"
            )

        all_cells: List[Dict[str, Any]] = []
        cumulative_spent = prior_spent
        aborted_reason: Optional[str] = None

        for w in waves:
            if cumulative_spent + w["est_pair_usd"] > max_budget_usd:
                aborted_reason = (
                    f"BUDGET GUARD STOP before {w['size']}/{w['quant']}: "
                    f"spent ${cumulative_spent:.2f} + next est ${w['est_pair_usd']:.2f} > cap ${max_budget_usd:.2f}"
                )
                print(aborted_reason)
                break

            fn = eval_quant_cell_l40s if w["gpu"] == "L40S" else eval_quant_cell_h100
            jobs = [
                (w["base_model"], w["size"], "base", w["quant"], n_gsm8k, n_math, max_tokens, seed),
                (w["wsp_model"], w["size"], "wsp", w["quant"], n_gsm8k, n_math, max_tokens, seed),
            ]
            pair_out = list(fn.starmap(jobs, return_exceptions=True))
            for job_args, raw_rec in zip(jobs, pair_out):
                if isinstance(raw_rec, BaseException):
                    worst_sec = float(TIMEOUT_SEC_BY_GPU[w["gpu"]])
                    worst_cost = round((worst_sec / 3600.0) * GPU_RATES_PER_HR[w["gpu"]], 4)
                    rec: Dict[str, Any] = {
                        "status": "container_exception",
                        "error": f"{type(raw_rec).__name__}: {raw_rec}",
                        "size": job_args[1],
                        "side": job_args[2],
                        "quant": job_args[3],
                        "model_id": job_args[0],
                        "gpu": w["gpu"],
                        "max_tokens": max_tokens,
                        "elapsed_sec": worst_sec,
                        "est_cost_usd": worst_cost,
                        "metrics": {},
                    }
                else:
                    rec = raw_rec
                all_cells.append(rec)
                _append_local_spend(rec)
                cumulative_spent += float(rec.get("est_cost_usd", 0.0))
                print(
                    f"  [{rec['status']}] {rec['size']} {rec['quant']:9s} {rec['side']:4s} "
                    f"elapsed={rec['elapsed_sec']}s cost=${rec['est_cost_usd']:.4f} "
                    f"(cum=${cumulative_spent:.4f}/${max_budget_usd:.2f})"
                )

            if cumulative_spent >= max_budget_usd:
                aborted_reason = f"BUDGET GUARD REACHED: spent ${cumulative_spent:.2f} >= cap ${max_budget_usd:.2f}"
                print(aborted_reason)
                break

        run_cost = round(sum(float(c.get("est_cost_usd", 0.0)) for c in all_cells), 4)
        payload = {
            "config": {
                "sizes": size_list,
                "quants": quant_list,
                "n_gsm8k": n_gsm8k,
                "n_math": n_math,
                "max_tokens": max_tokens,
                "seed": seed,
                "max_budget_usd": max_budget_usd,
                "ephemeral_only": True,
                "modal_profile": active_profile,
            },
            "run_est_cost_usd": run_cost,
            "cumulative_ledger_usd": round(cumulative_spent, 4),
            "aborted_reason": aborted_reason,
            "retention_summary": _summarize_retention(all_cells),
            "cells": all_cells,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / out_name
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved GPU audit results -> {out_path}")
        print(json.dumps(payload["retention_summary"], indent=2))


# ---------------------------------------------------------------------------
# Standalone CLI for $0 CPU SNR Audit & Offline Cost Plan Inspection
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Quantization Integrity Audit (CPU SNR + Plan Preview)")
    parser.add_argument("--cpu-snr-only", action="store_true", help="Run $0 CPU shard-level SNR audit locally")
    parser.add_argument("--plan-only", action="store_true", help="Print GPU wave plan & cost estimate ($0)")
    parser.add_argument("--sizes", default="2B,9B", help="Comma-separated model sizes (2B,9B)")
    parser.add_argument("--quants", default="bf16,fp8,bnb_4bit,w4a16_rtn", help="Comma-separated quant modes")
    parser.add_argument("--gpu-9b", default="H100", choices=list(GPU_RATES_PER_HR), help="GPU for 9B cells")
    args = parser.parse_args()

    size_list = [s.strip().upper() for s in args.sizes.split(",") if s.strip()]
    quant_list = [q.strip().lower() for q in args.quants.split(",") if q.strip()]
    for sz in size_list:
        if sz not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown size '{sz}'. Valid: {list(MODEL_REGISTRY)}")

    if args.cpu_snr_only:
        run_cpu_snr_audit(size_list)
        return

    for q in quant_list:
        if q not in SUPPORTED_QUANTS:
            raise SystemExit(f"Unknown quant '{q}'. Valid: {SUPPORTED_QUANTS}")
    est_total, waves = estimate_plan_cost(size_list, quant_list, gpu_override_9b=args.gpu_9b)
    print(f"Estimated GPU cost ({size_list} x {quant_list}): ${est_total:.4f} (hard cap ${HARD_BUDGET_CAP_USD:.2f})")
    for w in waves:
        print(f"  {w['size']} {w['quant']:10s} [{w['gpu']}] -> est ${w['est_pair_usd']:.4f} (2x {w['est_min_per_cell']}m)")
    if not args.plan_only:
        print(
            f"\nTo execute on Modal:\n"
            f"  modal run quant_audit.py "
            f"--sizes {args.sizes} --quants {args.quants}"
        )


if __name__ == "__main__":
    _cli_main()
