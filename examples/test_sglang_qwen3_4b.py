"""Smoke-test the sglang-hosted Qwen/Qwen3-4B server via its OpenAI-compatible API.

Prereqs:
    ./scripts/serve_qwen3_4b.sh   # start sglang on http://0.0.0.0:30000

Run:
    uv run python examples/test_sglang_qwen3_4b.py
    BASE_URL=http://localhost:30000/v1 MODEL=Qwen/Qwen3-4B \
        uv run python examples/test_sglang_qwen3_4b.py
"""

from __future__ import annotations

import os
import re
import sys
import time

from openai import OpenAI

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", flags=re.DOTALL)


def split_think(text: str | None) -> tuple[str, str]:
    """Fallback splitter for servers without a reasoning-parser.

    Returns (thinking, answer). If no <think> block is present, thinking is "".
    """
    if not text:
        return "", ""
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def main() -> int:
    base_url = os.environ.get("BASE_URL", "http://localhost:30000/v1")
    api_key = os.environ.get("API_KEY", "EMPTY")
    model = os.environ.get("MODEL", "Qwen/Qwen3-4B")

    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"[test] base_url = {base_url}")
    print(f"[test] model    = {model}")

    print("\n[test] GET /models")
    try:
        models = client.models.list()
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as exc:
        print(f"  failed: {exc}")
        return 1

    prompt = "In one short sentence, what is the capital of France?"
    print(f"\n[test] chat.completions (non-streaming)\n  prompt: {prompt}")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=8192,
    )
    dt = time.perf_counter() - t0
    msg = resp.choices[0].message
    content = msg.content
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None:
        reasoning, content = split_think(content)
    usage = resp.usage
    print(f"  thinking: {reasoning or '(none)'}")
    print(f"  response: {content}")
    print(f"  latency : {dt:.2f}s")
    if usage is not None:
        print(
            f"  tokens  : prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )

    print("\n[test] chat.completions (streaming)")
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Count from 1 to 5, space-separated."},
        ],
        temperature=0.8,
        max_tokens=8192,
        stream=True,
    )
    think_parts: list[str] = []
    answer_parts: list[str] = []
    first_token_at: float | None = None
    print("  [thinking] ", end="", flush=True)
    printing_think = True
    for chunk in stream:
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning_content", None) or ""
        c = delta.content or ""
        if (r or c) and first_token_at is None:
            first_token_at = time.perf_counter() - t0
        if r:
            think_parts.append(r)
            sys.stdout.write(r)
            sys.stdout.flush()
        if c:
            if printing_think:
                sys.stdout.write("\n  [answer]   ")
                sys.stdout.flush()
                printing_think = False
            answer_parts.append(c)
            sys.stdout.write(c)
            sys.stdout.flush()
    total = time.perf_counter() - t0
    print()
    thinking_text = "".join(think_parts)
    answer_text = "".join(answer_parts)
    if not thinking_text and answer_text:
        # Server didn't split; try regex fallback on the combined stream.
        thinking_text, answer_text = split_think(answer_text)
    print(f"  ttft    : {first_token_at:.2f}s" if first_token_at else "  ttft    : n/a")
    print(f"  total   : {total:.2f}s")
    print(f"  thinking: {thinking_text!r}")
    print(f"  answer  : {answer_text!r}")

    print("\n[test] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
