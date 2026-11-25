"""Self-contained grading utilities for GSM8K, MATH-500, IFEval, and loop detection."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# --- Thinking tag & repetition-loop detection ---

def strip_thinking(text: str, prompt_opened_think: bool = True) -> Tuple[str, str, bool]:
    """Return (thinking_part, answer_part, closed_thinking).

    Note: Qwen3.5 / Qwen3 chat templates with enable_thinking=True end the prompt
    with `<think>\\n`, so generated completions do not contain the opening tag;
    if `</think>` never appears when prompt_opened_think=True, thinking never closed.
    """
    if "</think>" in text:
        idx = text.rfind("</think>")
        think_part = text[:idx]
        if think_part.startswith("<think>"):
            think_part = think_part[len("<think>"):]
        return think_part.strip(), text[idx + len("</think>"):].strip(), True
    if "<think>" in text or prompt_opened_think:
        t = text[len("<think>"):].strip() if text.startswith("<think>") else text.strip()
        return t, "", False
    return "", text.strip(), True


def detect_repetition_loop(text: str, min_chunk: int = 32, min_repeats: int = 4) -> Dict[str, Any]:
    """Detect degenerate repetition loops in generated text."""
    if not text or len(text) < min_chunk * min_repeats:
        return {"is_loop": False, "max_repeat": 0, "tail_unique_ratio": 1.0}

    # Check tail window (last 2000 chars) for repeated non-overlapping lines or n-grams
    tail = text[-2500:]
    # 1. Word 8-gram repetition check
    words = tail.split()
    max_ngram_rep = 0
    if len(words) >= 32:
        ngrams = [" ".join(words[i : i + 8]) for i in range(0, len(words) - 7)]
        if ngrams:
            max_ngram_rep = Counter(ngrams).most_common(1)[0][1]

    # 2. Repeated substring / line check
    lines = [ln.strip() for ln in tail.splitlines() if len(ln.strip()) >= 16]
    max_line_rep = Counter(lines).most_common(1)[0][1] if lines else 0

    # 3. Character chunk uniqueness in tail
    chunks = [tail[i : i + min_chunk] for i in range(0, len(tail) - min_chunk + 1, min_chunk)]
    unique_ratio = (len(set(chunks)) / len(chunks)) if chunks else 1.0

    is_loop = (max_ngram_rep >= min_repeats) or (max_line_rep >= min_repeats) or (len(chunks) >= 12 and unique_ratio < 0.35)
    return {
        "is_loop": bool(is_loop),
        "max_repeat": int(max(max_ngram_rep, max_line_rep)),
        "tail_unique_ratio": round(float(unique_ratio), 4),
    }


# --- Math extraction & grading (GSM8K & MATH-500) ---

def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\fbox{")
        if idx == -1:
            return None
        start = idx + len("\\fbox{")
    else:
        start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_gsm8k_gold(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def extract_math_prediction(text: str, prefer_after_think: bool = True, prompt_opened_think: bool = True) -> Optional[str]:
    """Extract final answer from model output."""
    _, ans_part, closed = strip_thinking(text, prompt_opened_think=prompt_opened_think)
    search_text = ans_part if (prefer_after_think and closed and ans_part) else text

    boxed = extract_boxed(search_text)
    if boxed is not None:
        return boxed
    if search_text is not text:
        boxed = extract_boxed(text)
        if boxed is not None:
            return boxed

    # Fallback: "#### <num>" or "final answer is <...>"
    m = re.search(r"####\s*([^\n]+)", search_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer is|answer is|answer:)\s*\$?([^\n\.\$]+)", search_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last number fallback ONLY if thinking actually closed cleanly
    if closed and ans_part:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", ans_part)
        if nums:
            return nums[-1]
    return None


def _normalize_math_str(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\text{", "{").replace("\\mathrm{", "{").replace("\\mathbf{", "{")
    s = re.sub(r"\\dfrac\b", r"\\frac", s)
    s = re.sub(r"\\tfrac\b", r"\\frac", s)
    s = s.rstrip(".")
    # Remove trailing units/commas in plain numbers like 1,234
    s_nocomma = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return re.sub(r"\s+", "", s_nocomma)


def _try_parse_number(s: str) -> Optional[float]:
    s = s.strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def math_answers_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    p = _normalize_math_str(pred)
    g = _normalize_math_str(gold)
    if not p or not g:
        return False
    if p == g or p.lower() == g.lower():
        return True

    # Strip outer parens/braces
    p_strip = p.strip("{}")
    g_strip = g.strip("{}")
    if p_strip == g_strip:
        return True

    # Numeric comparison
    pn = _try_parse_number(p_strip)
    gn = _try_parse_number(g_strip)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= 1e-4 * max(1.0, abs(gn))

    # SymPy symbolic equivalence (best effort)
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        def _to_sym(expr_str: str):
            try:
                return parse_latex(expr_str)
            except Exception:
                return sympy.sympify(expr_str.replace("^", "**"))

        sp = _to_sym(p_strip)
        sg = _to_sym(g_strip)
        diff = sympy.simplify(sp - sg)
        if diff == 0:
            return True
        try:
            if abs(complex(diff.evalf())) < 1e-4:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


# --- Strict IFEval Instruction Verifier (Google IFEval 25-instruction suite) ---

def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    sents = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if s.strip()]
    return len(sents)


def _count_paragraphs(text: str) -> int:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paras)


def _check_relation(actual: int, relation: str, threshold: int) -> bool:
    rel = (relation or "").lower().strip()
    if rel in ("at least", ">="):
        return actual >= threshold
    if rel in ("less than", "<"):
        return actual < threshold
    if rel in ("at most", "<="):
        return actual <= threshold
    if rel in ("more than", ">"):
        return actual > threshold
    return actual >= threshold


def verify_ifeval_instruction(response: str, instruction_id: str, kwargs: Dict[str, Any]) -> bool:
    """Verify a single google/IFEval strict instruction on final (post-think) response."""
    text = response or ""
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}

    if instruction_id == "punctuation:no_comma":
        return "," not in text

    if instruction_id == "keywords:existence":
        kws = kw.get("keywords") or []
        tl = text.lower()
        return all(str(k).lower() in tl for k in kws)

    if instruction_id == "keywords:frequency":
        word = str(kw.get("keyword", "")).lower()
        freq = int(kw.get("frequency", 1))
        rel = str(kw.get("relation", "at least"))
        cnt = len(re.findall(rf"\b{re.escape(word)}\b", text.lower())) if word else 0
        return _check_relation(cnt, rel, freq)

    if instruction_id == "keywords:forbidden_words":
        fws = kw.get("forbidden_words") or []
        tl = text.lower()
        return all(not re.search(rf"\b{re.escape(str(w).lower())}\b", tl) for w in fws)

    if instruction_id == "keywords:letter_frequency":
        letter = str(kw.get("letter", "")).lower()
        let_freq = int(kw.get("let_frequency", 1))
        rel = str(kw.get("let_relation", "at least"))
        cnt = text.lower().count(letter) if letter else 0
        return _check_relation(cnt, rel, let_freq)

    if instruction_id == "length_constraints:number_words":
        n = int(kw.get("num_words", 0))
        rel = str(kw.get("relation", "at least"))
        return _check_relation(_count_words(text), rel, n)

    if instruction_id == "length_constraints:number_sentences":
        n = int(kw.get("num_sentences", 0))
        rel = str(kw.get("relation", "at least"))
        return _check_relation(_count_sentences(text), rel, n)

    if instruction_id == "length_constraints:number_paragraphs":
        n = int(kw.get("num_paragraphs", 1))
        # IFEval uses *** separator for paragraphs
        parts = [p.strip() for p in text.split("***") if p.strip()]
        return len(parts) == n

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        n_paras = int(kw.get("num_paragraphs", 1))
        nth = int(kw.get("nth_paragraph", 1))
        first_word = str(kw.get("first_word", "")).strip().lower()
        paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        if len(paras) != n_paras or nth < 1 or nth > len(paras):
            return False
        m = re.match(r"[^\w]*(\w+)", paras[nth - 1])
        return bool(m and m.group(1).lower() == first_word)

    if instruction_id == "detectable_content:number_placeholders":
        n = int(kw.get("num_placeholders", 1))
        found = re.findall(r"\[[^\]]+\]", text)
        return len(found) >= n

    if instruction_id == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker", "P.S.")).strip()
        pat = re.escape(marker).replace(r"\.", r"\.?\s*")
        return bool(re.search(pat, text, re.IGNORECASE))

    if instruction_id == "detectable_format:number_bullet_lists":
        n = int(kw.get("num_bullets", 1))
        bullets = re.findall(r"(?m)^\s*(?:\*|-)\s+\S+", text)
        return len(bullets) == n

    if instruction_id == "detectable_format:constrained_response":
        options = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
        t = text.strip()
        return any(opt in t for opt in options)

    if instruction_id == "detectable_format:number_highlighted_sections":
        n = int(kw.get("num_highlights", 1))
        highlights = re.findall(r"\*[^*\n]+\*", text)
        return len(highlights) >= n

    if instruction_id == "detectable_format:multiple_sections":
        splitter = str(kw.get("section_spliter", "Section")).strip()
        n = int(kw.get("num_sections", 1))
        matches = re.findall(rf"\b{re.escape(splitter)}\s+\d+", text, re.IGNORECASE)
        return len(matches) >= n

    if instruction_id == "detectable_format:json_format":
        t = text.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
        cand = m.group(1).strip() if m else t
        try:
            json.loads(cand)
            return True
        except Exception:
            return False

    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>\n]+>>", text))

    if instruction_id == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if instruction_id == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)

    if instruction_id == "change_case:capital_word_frequency":
        freq = int(kw.get("capital_frequency", 1))
        rel = str(kw.get("capital_relation", "at least"))
        cap_words = [w for w in re.findall(r"\b[A-Za-z]+\b", text) if w.isupper()]
        return _check_relation(len(cap_words), rel, freq)

    if instruction_id == "startend:end_checker":
        phrase = str(kw.get("end_phrase", "")).strip()
        return text.strip().rstrip('"').rstrip("'").lower().endswith(phrase.lower())

    if instruction_id == "startend:quotation":
        t = text.strip()
        return len(t) >= 2 and t[0] == '"' and t[-1] == '"'

    if instruction_id == "combination:two_responses":
        parts = [p.strip() for p in text.split("******") if p.strip()]
        return len(parts) == 2 and parts[0].lower() != parts[1].lower()

    if instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = str(kw.get("prompt_to_repeat", "")).strip()
        return text.strip().lower().startswith(prompt_to_repeat.lower())

    if instruction_id == "language:response_language":
        lang = str(kw.get("language", "")).strip().lower()
        if not text.strip():
            return False
        try:
            import langdetect

            detected = langdetect.detect(text)
            return detected.lower().startswith(lang)
        except Exception:
            return False

    return False


def grade_ifeval_sample(
    response_raw: str,
    instruction_ids: List[str],
    kwargs_list: List[Dict[str, Any]],
    prompt_opened_think: bool = True,
) -> Dict[str, Any]:
    """Grade one IFEval sample after stripping <think>...</think>."""
    _, ans_part, closed = strip_thinking(response_raw, prompt_opened_think=prompt_opened_think)
    final_resp = ans_part if closed else ""
    results = [
        verify_ifeval_instruction(final_resp, iid, kw)
        for iid, kw in zip(instruction_ids, kwargs_list)
    ]
    return {
        "prompt_strict": bool(results and all(results)),
        "inst_passed": int(sum(1 for r in results if r)),
        "inst_total": int(len(results)),
        "closed_thinking": closed,
    }
