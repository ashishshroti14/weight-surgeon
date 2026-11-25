"""Harder & Out-of-Distribution (OOD) Stress-Test Suite for Qwen3.5 Base vs WSP-Patched models.

Runs directly from Hugging Face Hub on ephemeral Modal H100 GPUs (zero persistent volume needed).
Evaluates:
  1. MATH-500 (n=500) under Qwen's official recommended sampling (temp=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=8192)
  2. AIME 2024 + 2025 (n=60) both Greedy (temp=0, pp=0) and Official Sampling (max_tokens=8192)
  3. GPQA Diamond (n=198, PhD science, 100% OOD) Greedy (max_tokens=8192)
  4. MMLU-Pro (n=420 stratified across all 14 domains, 10-choice, 100% OOD) in BOTH Non-Thinking (collateral-damage check) and Thinking modes
  5. HumanEval (n=164 Python code generation, real subprocess unit-test pass@1, 100% OOD) Greedy (max_tokens=4096)
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

APP_NAME = "wsp-harder-evals"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "transformers",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "langdetect",
        "einops",
        "pillow",
        "qwen-vl-utils",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_file("eval_utils.py", "/root/eval_utils.py")
)


def _apply_chat(tokenizer: Any, content: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": content}]
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


def _extract_mc_choice(text: str, valid_letters: str, prompt_opened_think: bool = True) -> Optional[str]:
    from eval_utils import extract_boxed, strip_thinking

    _, after, closed = strip_thinking(text, prompt_opened_think=prompt_opened_think)
    target = after if (closed and after.strip()) else text

    b = extract_boxed(target)
    if b:
        m = re.search(rf"\b([{valid_letters}])\b", b.upper())
        if m:
            return m.group(1)
        m2 = re.search(rf"([{valid_letters}])", b.upper())
        if m2:
            return m2.group(1)

    patterns = [
        rf"(?:final\s+)?answer\s*(?:is|:)\s*\(?([{valid_letters}])\)?\b",
        rf"\boption\s*\(?([{valid_letters}])\)?\b",
        rf"\*\*([{valid_letters}])\*\*",
        rf"^\s*\(?([{valid_letters}])\)?[\.\):]?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, target, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    # Last standalone valid letter near the end
    tail = target[-200:]
    matches = re.findall(rf"\b([{valid_letters}])\b", tail)
    if matches:
        return matches[-1].upper()
    return None


def _extract_python_code(text: str, prompt_code: str, entry_point: str, prompt_opened_think: bool = True) -> str:
    from eval_utils import strip_thinking

    _, after, closed = strip_thinking(text, prompt_opened_think=prompt_opened_think)
    target = after if (closed and after.strip()) else text

    # Prefer fenced python blocks (take the last one containing def entry_point, else last python block)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", target, flags=re.DOTALL | re.IGNORECASE)
    chosen = ""
    if blocks:
        for blk in reversed(blocks):
            if f"def {entry_point}" in blk:
                chosen = blk
                break
        if not chosen:
            chosen = blocks[-1]
    else:
        chosen = target

    chosen = chosen.strip("\n")
    if f"def {entry_point}" in chosen:
        # Keep any imports from prompt_code that might be needed before the function def
        prefix_lines = []
        for ln in prompt_code.splitlines():
            if ln.strip().startswith(("from ", "import ")):
                prefix_lines.append(ln)
        prefix = "\n".join(prefix_lines)
        return (prefix + "\n\n" + chosen) if prefix else chosen
    # Otherwise treat as function body continuation
    if chosen and not chosen.startswith((" ", "\t")):
        chosen = "\n".join("    " + ln if ln.strip() else ln for ln in chosen.splitlines())
    return prompt_code + "\n" + chosen


def _run_humaneval_test(candidate_code: str, test_code: str, entry_point: str, timeout_sec: float = 5.0) -> bool:
    script = f"{candidate_code}\n\n{test_code}\n\ncheck({entry_point})\n"
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def _load_harder_datasets() -> Dict[str, Any]:
    from datasets import load_dataset

    # 1. MATH-500
    math500 = [
        {"problem": r["problem"], "answer": r["answer"]}
        for r in load_dataset("HuggingFaceH4/MATH-500", split="test")
    ]

    # 2. AIME 2024 + AIME 2025 (n=60)
    aime24 = [
        {"problem": r["problem"], "answer": str(r["answer"]).strip(), "year": 2024}
        for r in load_dataset("HuggingFaceH4/aime_2024", split="train")
    ]
    aime25 = [
        {"problem": r["problem"], "answer": str(r["answer"]).strip(), "year": 2025}
        for r in load_dataset("MathArena/aime_2025", split="train")
    ]
    aime_all = aime24 + aime25

    # 3. GPQA Diamond (n=198, deterministic choice shuffling)
    gpqa_raw = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train")
    gpqa_items = []
    letters4 = "ABCD"
    for idx, r in enumerate(gpqa_raw):
        opts = [
            (str(r["Correct Answer"]).strip(), True),
            (str(r["Incorrect Answer 1"]).strip(), False),
            (str(r["Incorrect Answer 2"]).strip(), False),
            (str(r["Incorrect Answer 3"]).strip(), False),
        ]
        rng = random.Random(42 + idx)
        rng.shuffle(opts)
        gold = next(letters4[i] for i, (_, ok) in enumerate(opts) if ok)
        gpqa_items.append(
            {
                "question": str(r["Question"]).strip(),
                "options": [txt for txt, _ in opts],
                "answer": gold,
                "domain": str(r.get("High-level domain", "")),
            }
        )

    # 4. MMLU-Pro stratified (30 per category across all 14 categories = 420)
    mmlu_raw = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in mmlu_raw:
        by_cat[str(r["category"])].append(
            {
                "question": str(r["question"]).strip(),
                "options": list(r["options"]),
                "answer": str(r["answer"]).strip().upper(),
                "category": str(r["category"]),
            }
        )
    mmlu_items = []
    for cat_idx, cat in enumerate(sorted(by_cat.keys())):
        rows = by_cat[cat]
        rng = random.Random(100 + cat_idx)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        for i in idxs[:30]:
            mmlu_items.append(rows[i])

    # 5. HumanEval (n=164)
    humaneval_items = [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in load_dataset("openai/openai_humaneval", "openai_humaneval", split="test")
    ]

    return {
        "math500": math500,
        "aime": aime_all,
        "gpqa": gpqa_items,
        "mmlu_pro": mmlu_items,
        "humaneval": humaneval_items,
    }


@app.function(image=image, gpu="H100", cpu=4, memory=24576, timeout=5400)
def eval_model_harder(model_id: str, tag: str) -> Dict[str, Any]:
    import gc
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
    ds = _load_harder_datasets()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=12288,
        trust_remote_code=True,
        enforce_eager=False,
    )

    sp_greedy_8k = SamplingParams(
        temperature=0.0, top_p=1.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=8192
    )
    sp_greedy_4k = SamplingParams(
        temperature=0.0, top_p=1.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=4096
    )
    sp_greedy_nt = SamplingParams(
        temperature=0.0, top_p=1.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=512
    )
    # Qwen3.5 official recommended thinking sampling params
    sp_qwen_rec_8k = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=8192, seed=42
    )

    def _run_math(items: List[Dict[str, Any]], sp: SamplingParams) -> Dict[str, Any]:
        prompts = [
            _apply_chat(
                tokenizer,
                f"{it['problem'].strip()}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                enable_thinking=True,
            )
            for it in items
        ]
        outs = llm.generate(prompts, sp)
        correct = trunc = loops = closed_n = toks = 0
        for it, p_str, out in zip(items, prompts, outs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            toks += len(comp.token_ids)
            if comp.finish_reason == "length":
                trunc += 1
            if detect_repetition_loop(txt)["is_loop"]:
                loops += 1
            _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
            if closed:
                closed_n += 1
            pred = extract_math_prediction(txt, prefer_after_think=True, prompt_opened_think=opened)
            if math_answers_equal(pred, it["answer"]):
                correct += 1
        n = len(items)
        return {
            "n": n,
            "accuracy": round(100.0 * correct / n, 2),
            "correct": correct,
            "truncation_pct": round(100.0 * trunc / n, 2),
            "loop_pct": round(100.0 * loops / n, 2),
            "closed_think_pct": round(100.0 * closed_n / n, 2),
            "avg_tokens": round(toks / n, 1),
        }

    def _run_mc(items: List[Dict[str, Any]], valid_letters: str, sp: SamplingParams, enable_thinking: bool) -> Dict[str, Any]:
        prompts = []
        for it in items:
            opts_str = "\n".join(f"({valid_letters[i]}) {opt}" for i, opt in enumerate(it["options"]))
            msg = (
                f"{it['question']}\n\nOptions:\n{opts_str}\n\n"
                f"Select the single best option letter ({valid_letters[0]}-{valid_letters[len(it['options'])-1]}) "
                "and put your final choice letter inside \\boxed{}."
            )
            prompts.append(_apply_chat(tokenizer, msg, enable_thinking=enable_thinking))
        outs = llm.generate(prompts, sp)
        correct = trunc = loops = closed_n = toks = 0
        for it, p_str, out in zip(items, prompts, outs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            toks += len(comp.token_ids)
            if comp.finish_reason == "length":
                trunc += 1
            if detect_repetition_loop(txt)["is_loop"]:
                loops += 1
            _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
            if closed:
                closed_n += 1
            pred = _extract_mc_choice(txt, valid_letters[: len(it["options"])], prompt_opened_think=opened)
            if pred == it["answer"]:
                correct += 1
        n = len(items)
        return {
            "n": n,
            "accuracy": round(100.0 * correct / n, 2),
            "correct": correct,
            "truncation_pct": round(100.0 * trunc / n, 2),
            "loop_pct": round(100.0 * loops / n, 2),
            "closed_think_pct": round(100.0 * closed_n / n, 2),
            "avg_tokens": round(toks / n, 1),
        }

    def _run_humaneval(items: List[Dict[str, Any]], sp: SamplingParams) -> Dict[str, Any]:
        prompts = [
            _apply_chat(
                tokenizer,
                "Complete the following Python function. Provide the complete Python function inside a ```python``` code block.\n\n"
                f"```python\n{it['prompt']}```",
                enable_thinking=True,
            )
            for it in items
        ]
        outs = llm.generate(prompts, sp)
        passed = trunc = loops = closed_n = toks = 0
        for it, p_str, out in zip(items, prompts, outs):
            comp = out.outputs[0]
            txt = comp.text
            opened = "<think>" in p_str
            toks += len(comp.token_ids)
            if comp.finish_reason == "length":
                trunc += 1
            if detect_repetition_loop(txt)["is_loop"]:
                loops += 1
            _, _, closed = strip_thinking(txt, prompt_opened_think=opened)
            if closed:
                closed_n += 1
            code = _extract_python_code(txt, it["prompt"], it["entry_point"], prompt_opened_think=opened)
            if _run_humaneval_test(code, it["test"], it["entry_point"]):
                passed += 1
        n = len(items)
        return {
            "n": n,
            "pass_at_1": round(100.0 * passed / n, 2),
            "passed": passed,
            "truncation_pct": round(100.0 * trunc / n, 2),
            "loop_pct": round(100.0 * loops / n, 2),
            "closed_think_pct": round(100.0 * closed_n / n, 2),
            "avg_tokens": round(toks / n, 1),
        }

    metrics = {
        "math500_qwen_sampling_8k": _run_math(ds["math500"], sp_qwen_rec_8k),
        "aime24_25_greedy_8k": _run_math(ds["aime"], sp_greedy_8k),
        "aime24_25_qwen_sampling_8k": _run_math(ds["aime"], sp_qwen_rec_8k),
        "gpqa_diamond_greedy_8k": _run_mc(ds["gpqa"], "ABCD", sp_greedy_8k, enable_thinking=True),
        "mmlu_pro_nonthinking": _run_mc(ds["mmlu_pro"], "ABCDEFGHIJ", sp_greedy_nt, enable_thinking=False),
        "mmlu_pro_thinking_greedy_4k": _run_mc(ds["mmlu_pro"], "ABCDEFGHIJ", sp_greedy_4k, enable_thinking=True),
        "humaneval_greedy_4k": _run_humaneval(ds["humaneval"], sp_greedy_4k),
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    est_cost = (elapsed / 3600.0) * 4.17
    return {
        "tag": tag,
        "model_id": model_id,
        "elapsed_sec": round(elapsed, 1),
        "est_cost_usd": round(est_cost, 4),
        "metrics": metrics,
    }


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-9B",
    wsp_model: str = "ashishshroti14/Qwen3.5-9B-WSP",
    out_name: str = "harder_Qwen3.5-9B-WSP.json",
):
    jobs = [(base_model, "base"), (wsp_model, "patched")]
    outs = list(eval_model_harder.starmap(jobs))
    res = {
        "base_model": base_model,
        "wsp_model": wsp_model,
        "base": outs[0],
        "patched": outs[1],
        "total_est_cost_usd": round(outs[0]["est_cost_usd"] + outs[1]["est_cost_usd"], 4),
    }
    out_path = Path("results") / out_name
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
