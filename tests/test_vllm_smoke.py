"""
tests/test_vllm_smoke.py
========================
Self-contained vLLM smoke test. Exercises every request path used by
Phase 0R against a live vLLM server (booted by scripts/submit_vllm_smoke.sh).
No crops, no CSV; just synthetic images and short prompts.

Usage:
    VLLM_BASE_URL=http://localhost:8000/v1 python tests/test_vllm_smoke.py

Exit code 0 = all PASS, 1 = any FAIL.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Tuple

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.vllm_client import VLLMClient  # noqa: E402

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen2.5-VL-7B-Instruct")
TIMEOUT  = int(os.environ.get("VLLM_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------

def _synthetic_image(fmt: str, size: int = 256) -> bytes:
    from PIL import Image, ImageDraw  # local import — heavy dep
    img = Image.new("RGB", (size, size), (40, 120, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4],
                   fill=(180, 60, 50))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _data_url(b: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")


# ---------------------------------------------------------------------------
# Test cases — each returns (name, ok, detail)
# ---------------------------------------------------------------------------

def t_server_health() -> Tuple[bool, str]:
    r = requests.get(f"{BASE_URL}/models", timeout=10)
    r.raise_for_status()
    ids = [m.get("id") for m in r.json().get("data", [])]
    if MODEL not in ids:
        return False, f"served ids={ids}, expected {MODEL}"
    return True, f"served {MODEL}"


def t_chat_text_only() -> Tuple[bool, str]:
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=32, timeout=TIMEOUT)
    c.chat_request_logprobs = False
    text, n = c.chat(messages=[{"role": "user", "content": "Reply with the word OK."}])
    if not text.strip():
        return False, "empty response"
    return True, f"got {n} tokens, first={text.strip()[:40]!r}"


def t_chat_with_jpeg() -> Tuple[bool, str]:
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=48, timeout=TIMEOUT)
    c.chat_request_logprobs = False
    url = _data_url(_synthetic_image("JPEG"), "image/jpeg")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Describe the dominant colors in one short sentence."},
    ]}]
    text, n = c.chat(messages=msgs)
    if not text.strip():
        return False, "empty response"
    return True, f"{n} tokens, {text.strip()[:60]!r}"


def t_chat_with_png() -> Tuple[bool, str]:
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=48, timeout=TIMEOUT)
    c.chat_request_logprobs = False
    url = _data_url(_synthetic_image("PNG"), "image/png")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Is there a red square? Answer yes or no."},
    ]}]
    text, n = c.chat(messages=msgs)
    if not text.strip():
        return False, "empty response"
    return True, f"{n} tokens, {text.strip()[:60]!r}"


def t_chat_with_logprobs() -> Tuple[bool, str]:
    """Entropy-routing path: chat_request_logprobs=True."""
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=24, timeout=TIMEOUT)
    c.chat_request_logprobs = True
    url = _data_url(_synthetic_image("JPEG"), "image/jpeg")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "One word color."},
    ]}]
    r = c.chat_with_logprobs(messages=msgs)
    if r.content_logprobs is None or len(r.content_logprobs) == 0:
        return False, "no logprobs in response"
    return True, f"{len(r.content_logprobs)} logprob items, sample token={r.token_strings[:3]}"


def t_score_labels_vision() -> Tuple[bool, str]:
    """Structured-outputs vision scoring path."""
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=32, timeout=TIMEOUT)
    img_b64 = base64.b64encode(_synthetic_image("JPEG")).decode("ascii")
    scores = c.score_labels(
        prompt_prefix="What is the dominant color? Choose:",
        label_list=["red", "green", "blue"],
        image_b64=img_b64,
    )
    if set(scores.keys()) != {"red", "green", "blue"}:
        return False, f"label set mismatch: {scores}"
    total = sum(scores.values())
    if not (0.99 <= total <= 1.01):
        return False, f"scores don't sum to 1: {scores} sum={total}"
    return True, f"top={max(scores, key=scores.get)} dist={ {k: round(v,2) for k,v in scores.items()} }"


def t_score_labels_text_only() -> Tuple[bool, str]:
    """Text-only /completions scoring path."""
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=24, timeout=TIMEOUT)
    scores = c.score_labels(
        prompt_prefix="The opposite of hot is",
        label_list=["cold", "warm", "fire"],
        image_b64=None,
    )
    if not scores:
        return False, "empty scores"
    if not (0.99 <= sum(scores.values()) <= 1.01):
        return False, f"sum mismatch: {scores}"
    return True, f"top={max(scores, key=scores.get)} dist={ {k: round(v,2) for k,v in scores.items()} }"


def t_oversized_prompt_400_body() -> Tuple[bool, str]:
    """Push past max_model_len so vLLM returns 400; verify the body is
    surfaced in the HTTPError message (commit a541684)."""
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=16, timeout=TIMEOUT)
    c.chat_request_logprobs = False
    huge = "lorem ipsum dolor sit amet " * 20000  # ~100k tokens, definitely overflows
    try:
        c.chat(messages=[{"role": "user", "content": huge}])
    except requests.HTTPError as e:
        msg = str(e)
        if "::" not in msg or "400" not in msg:
            return False, f"error did not include body marker: {msg[:200]}"
        return True, f"400 with body: {msg.split('::', 1)[1][:120].strip()}"
    return False, "expected HTTPError, got success"


def t_concurrent_chat() -> Tuple[bool, str]:
    """Fire 8 chats in parallel — verifies continuous batching path."""
    c = VLLMClient(base_url=BASE_URL, model=MODEL, max_new_tokens=24, timeout=TIMEOUT)
    c.chat_request_logprobs = False
    url = _data_url(_synthetic_image("JPEG"), "image/jpeg")

    def _one(i: int) -> bool:
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": f"Reply with the number {i}."},
        ]}]
        t, _ = c.chat(messages=msgs, seed=i)
        return bool(t.strip())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in as_completed(pool.submit(_one, i) for i in range(8))]
    dt = time.time() - t0
    if not all(results):
        return False, f"only {sum(results)}/8 succeeded"
    return True, f"8/8 ok in {dt:.1f}s"


def t_cache_fetch_smoke() -> Tuple[bool, str]:
    """Validate ensure_state_image_cache.py end-to-end with a tiny CSV."""
    import csv as _csv
    import subprocess
    tmp_dir = Path("/tmp/vllm_smoke_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "tiny.csv"
    cache_dir = tmp_dir / "cache"
    # 2 rows, same (crop, disease, state), different Image Number — exercises dedup
    # and --all-rows. We use a well-known stable image from the actual CSV.
    src_csv = REPO / "BugWood_Diseases_usable.csv"
    if not src_csv.is_file():
        return False, f"missing {src_csv}"
    with open(src_csv, newline="") as fh:
        reader = _csv.DictReader(fh)
        head = next(reader)
        second = next(reader, None)
    if not head or not second:
        return False, "source CSV too short"
    fieldnames = list(head.keys())
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(head)
        w.writerow(second)
    cmd = [
        sys.executable, str(REPO / "scripts" / "ensure_state_image_cache.py"),
        "--csv", str(csv_path),
        "--cache-dir", str(cache_dir),
        "--all-rows", "--workers", "2",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return False, f"fetcher exit={out.returncode}: {out.stderr[:200]}"
    files = list(cache_dir.iterdir())
    if len(files) < 1:
        return False, f"no files in {cache_dir}: {out.stdout[-200:]}"
    return True, f"{len(files)} files cached, sample={files[0].name}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CASES: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
    ("server_health",            t_server_health),
    ("chat_text_only",           t_chat_text_only),
    ("chat_with_jpeg",           t_chat_with_jpeg),
    ("chat_with_png",            t_chat_with_png),
    ("chat_with_logprobs",       t_chat_with_logprobs),
    ("score_labels_vision",      t_score_labels_vision),
    ("score_labels_text_only",   t_score_labels_text_only),
    ("oversized_prompt_400_body", t_oversized_prompt_400_body),
    ("concurrent_chat_x8",       t_concurrent_chat),
    ("cache_fetch_smoke",        t_cache_fetch_smoke),
]


def main() -> int:
    print(f"vLLM smoke: BASE_URL={BASE_URL} MODEL={MODEL}")
    print("=" * 70)
    results: List[Tuple[str, bool, str, float]] = []
    for name, fn in CASES:
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:
            ok = False
            detail = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        dt = time.time() - t0
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:<28} ({dt:5.1f}s)  {detail}")
        results.append((name, ok, detail, dt))

    n_pass = sum(1 for _, ok, *_ in results if ok)
    n_fail = len(results) - n_pass
    print("=" * 70)
    print(f"  {n_pass}/{len(results)} PASSED, {n_fail} FAILED")

    # Machine-readable summary for sbatch log parsing.
    summary = {
        "passed": n_pass, "failed": n_fail,
        "cases": [{"name": n, "ok": ok, "detail": d, "elapsed_s": round(t, 2)}
                  for n, ok, d, t in results],
    }
    out_path = Path(os.environ.get("VLLM_SMOKE_RESULTS", "logs/vllm_smoke_results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  results → {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
