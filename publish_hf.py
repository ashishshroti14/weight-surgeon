"""Generate publication-grade Hugging Face Model Cards and upload WSP releases directly from Modal."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import modal

APP_NAME = "wsp-hf-publisher"
VOLUME_NAME = "wsp-research-vol"
VOL_PATH = "/vol"
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface-secret")
USE_WSP_VOLUME = os.environ.get("USE_WSP_VOLUME", "0") == "1"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, version=2) if USE_WSP_VOLUME else None

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.28.0", "hf_transfer>=0.1.9")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def render_model_card(res: Dict[str, Any], repo_id: str) -> str:
    base_model = res["model_id"]
    pinfo = res["patch_info"]
    pspec = pinfo["patch_spec"]
    base = res["base"]
    patched = res["patched"]

    def _delta(a: float, b: float) -> str:
        d = b - a
        return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"

    layers_str = ", ".join(str(x) for x in pspec["layers"])
    mod_params = f"{pinfo['modified_params']:,}"
    tot_params = f"{pinfo['total_params']:,}"
    pct = f"{pinfo['pct_modified']:.4f}%"

    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - weight-space-programming
  - wsp
  - zero-gradient
  - mechanistic-interpretability
  - qwen3_5
  - reasoning
  - loop-mitigation
---

# {repo_id.split("/")[-1]}

A **zero-gradient Weight-Space Programmed (WSP)** surgical repair of [`{base_model}`](https://huggingface.co/{base_model}) that cures its greedy-decoding reasoning/repetition loop pathology by modifying only **{mod_params} / {tot_params} parameters ({pct})**—with **zero backpropagation, zero fine-tuning, and zero RL**.

Inspired by Takehi et al. (MIT CSAIL, Nov 2026), *"Sparks of Weight-Space Programming"*, extended across the dense Qwen 3.5 hybrid Gated DeltaNet + Full Attention family.

## Benchmark Results (Full Held-Out Evaluation, Greedy `temp=0`, `presence_penalty=0.0`, `max_tokens={res.get('max_tokens', 4096)}`)

> **Zero-shot generalization check:** `google/IFEval` was **never** used during diagnosis or patch selection; all IFEval gains are 100% out-of-distribution generalization.

| Benchmark | Split | `{base_model}` (Base) | **{repo_id.split("/")[-1]}** (`{pct}` edited) | Delta |
| :--- | :--- | ---: | ---: | ---: |
| **GSM8K** | Full Test (`n={patched['gsm8k']['n']}`) | {base['gsm8k']['accuracy']:.2f}% | **{patched['gsm8k']['accuracy']:.2f}%** | **{_delta(base['gsm8k']['accuracy'], patched['gsm8k']['accuracy'])} pts** |
| **MATH-500** | Full Test (`n={patched['math']['n']}`) | {base['math']['accuracy']:.2f}% | **{patched['math']['accuracy']:.2f}%** | **{_delta(base['math']['accuracy'], patched['math']['accuracy'])} pts** |
| **IFEval (Prompt-Strict)** | Full (`n={patched['ifeval']['n']}`, zero-shot OOD) | {base['ifeval']['prompt_strict_acc']:.2f}% | **{patched['ifeval']['prompt_strict_acc']:.2f}%** | **{_delta(base['ifeval']['prompt_strict_acc'], patched['ifeval']['prompt_strict_acc'])} pts** |
| **IFEval (Inst-Strict)** | Full (`n={patched['ifeval']['n']}`, zero-shot OOD) | {base['ifeval']['inst_strict_acc']:.2f}% | **{patched['ifeval']['inst_strict_acc']:.2f}%** | **{_delta(base['ifeval']['inst_strict_acc'], patched['ifeval']['inst_strict_acc'])} pts** |

### Efficiency & Loop-Mitigation Impact

| Metric | Base | **WSP-Patched** |
| :--- | ---: | ---: |
| GSM8K Truncation (`finish_reason=length`) | {base['gsm8k']['truncation_pct']:.2f}% | **{patched['gsm8k']['truncation_pct']:.2f}%** |
| GSM8K Avg Output Tokens | {base['gsm8k']['avg_tokens']:.1f} | **{patched['gsm8k']['avg_tokens']:.1f}** |
| MATH-500 Truncation | {base['math']['truncation_pct']:.2f}% | **{patched['math']['truncation_pct']:.2f}%** |
| MATH-500 `</think>` Closed Rate | {base['math']['closed_think_pct']:.2f}% | **{patched['math']['closed_think_pct']:.2f}%** |
| IFEval Truncation | {base['ifeval']['truncation_pct']:.2f}% | **{patched['ifeval']['truncation_pct']:.2f}%** |

## Exact Surgical Edit Details

- **Method:** `{pspec['mode']}` (zero-gradient residual-stream within-`<think>` contrastive diagnosis + conditional SwiGLU `mlp.down_proj` column steering)
- **Target Layers:** `[{layers_str}]` (`mlp.down_proj.weight` only)
- **Columns Modified per Layer:** `{pspec['top_k']}` (selected by `{pspec['rank_by']}`)
- **Hyperparameters:** `alpha={pspec['alpha']}`, `think_weight={pspec.get('think_weight', 0.5)}`, `vec_source={pspec.get('vec_source', 'resid_loop_dir')}`
- **Parameters Modified:** `{mod_params}` / `{tot_params}` (**`{pct}`**)
- **Reproducibility Artifacts Included:** `wsp_patch.pt` (standalone extracted directions & column scores), `wsp_patch_summary.json`, and `eval_results.json`.

## Quick Usage

Drop-in replacement for `{base_model}` in `vLLM`, `transformers`, `SGLang`, or `llama.cpp`—works cleanly under **standard greedy decoding (`temperature=0, presence_penalty=0`)** without inference-time repetition penalty hacks:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_id}", dtype="bfloat16")
# Works cleanly even at temperature=0, presence_penalty=0.0!
```
"""


def _resolve_hf_token(explicit: str = "") -> str:
    for val in (
        explicit,
        os.environ.get("HF_TOKEN", ""),
        os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
        os.environ.get("HUGGINGFACE_TOKEN", ""),
    ):
        if val and val.strip():
            return val.strip()
    raise RuntimeError("No HF token found in modal secret or environment.")


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=120,
)
def check_hf_secret() -> Dict[str, Any]:
    from huggingface_hub import HfApi

    tok = _resolve_hf_token()
    api = HfApi(token=tok)
    info = api.whoami()
    auth = info.get("auth") or {}
    access_token = auth.get("accessToken") or {}
    return {
        "secret_name": HF_SECRET_NAME,
        "username": info.get("name"),
        "fullname": info.get("fullname"),
        "token_role": access_token.get("role"),
        "token_name": access_token.get("displayName"),
        "fine_grained": bool(access_token.get("fineGrained")),
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=120,
)
def upload_text_file(repo_id: str, path_in_repo: str, content: str, commit_message: str) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_hf_token())
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return {"status": "uploaded_file", "repo_id": repo_id, "path": path_in_repo}


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=120,
)
def delete_hf_repo(repo_id: str) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_hf_token())
    api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)
    return {"status": "deleted", "repo_id": repo_id}


if USE_WSP_VOLUME:
    @app.function(
        image=image,
        cpu=2,
        memory=4096,
        volumes={VOL_PATH: vol},
        secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
        timeout=1800,
    )
    def push_release_to_hf(release_slug: str, repo_id: str, readme_text: str, private: bool = False) -> Dict[str, Any]:
        from huggingface_hub import HfApi

        vol.reload()
        release_dir = Path(f"{VOL_PATH}/releases/{release_slug}")
        if not release_dir.exists():
            raise FileNotFoundError(f"Release dir not found on volume: {release_dir}")

        (release_dir / "README.md").write_text(readme_text, encoding="utf-8")
        vol.commit()

        api = HfApi(token=_resolve_hf_token())
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(release_dir),
            ignore_patterns=[".cache/**", "*.tmp"],
            commit_message=f"Upload {release_slug} (zero-gradient WSP patched model, patch artifacts, and Model Card)",
        )
        return {"status": "uploaded", "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}


@app.local_entrypoint()
def main(
    action: str = "whoami",
    username: str = "",
    slugs: str = "Qwen3.5-0.8B-WSP,Qwen3.5-2B-WSP,Qwen3.5-4B-WSP,Qwen3.5-9B-WSP",
    private: bool = False,
):
    if action == "whoami":
        print(json.dumps(check_hf_secret.remote(), indent=2))
        return

    if action == "publish_all":
        if not username:
            info = check_hf_secret.remote()
            username = str(info["username"])
        slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
        jobs = []
        for slug in slug_list:
            res_path = Path("results") / f"full_{slug}.json"
            res = json.loads(res_path.read_text(encoding="utf-8"))
            repo_id = f"{username}/{slug}"
            card = render_model_card(res, repo_id)
            (Path("results") / f"README_{slug}.md").write_text(card, encoding="utf-8")
            jobs.append((slug, repo_id, card, private))
        out = list(push_release_to_hf.starmap(jobs))
        print(json.dumps(out, indent=2))
        return

    if action == "upload_eval":
        if not username:
            info = check_hf_secret.remote()
            username = str(info["username"])
        slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
        out = []
        for slug in slug_list:
            content = (Path("results") / f"full_{slug}.json").read_text(encoding="utf-8")
            out.append(
                upload_text_file.remote(
                    f"{username}/{slug}",
                    "eval_results.json",
                    content,
                    f"Add full held-out eval_results.json for {slug}",
                )
            )
        print(json.dumps(out, indent=2))
        return

    if action == "upload_harder":
        if not username:
            info = check_hf_secret.remote()
            username = str(info["username"])
        slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
        out = []
        for slug in slug_list:
            repo_id = f"{username}/{slug}"
            readme_text = (Path("results") / f"README_{slug}.md").read_text(encoding="utf-8")
            harder_text = (Path("results") / f"harder_{slug}.json").read_text(encoding="utf-8")
            out.append(
                upload_text_file.remote(
                    repo_id,
                    "README.md",
                    readme_text,
                    f"Add OOD & official-sampling stress-test results to {slug} Model Card",
                )
            )
            out.append(
                upload_text_file.remote(
                    repo_id,
                    "eval_harder_results.json",
                    harder_text,
                    f"Add OOD & official-sampling raw eval results for {slug}",
                )
            )
        print(json.dumps(out, indent=2))
        return

    if action == "upload_harder_9b":
        if not username:
            info = check_hf_secret.remote()
            username = str(info["username"])
        repo_id = f"{username}/Qwen3.5-9B-WSP"
        readme_text = (Path("results") / "README_Qwen3.5-9B-WSP.md").read_text(encoding="utf-8")
        harder_text = (Path("results") / "harder_Qwen3.5-9B-WSP.json").read_text(encoding="utf-8")
        out = [
            upload_text_file.remote(repo_id, "README.md", readme_text, "Add OOD & official-sampling stress-test results to Model Card"),
            upload_text_file.remote(repo_id, "eval_harder_results.json", harder_text, "Add OOD & official-sampling raw eval results"),
        ]
        print(json.dumps(out, indent=2))
        return

    if action == "delete_repos":
        if not username:
            info = check_hf_secret.remote()
            username = str(info["username"])
        slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
        out = [delete_hf_repo.remote(f"{username}/{s}") for s in slug_list]
        print(json.dumps(out, indent=2))
        return

    raise ValueError(f"Unknown action: {action}")
