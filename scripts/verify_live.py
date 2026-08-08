#!/usr/bin/env python
"""Live end-to-end check against an OpenAI-compatible server (vLLM or OpenAI).

Contrasts a low-entropy prompt against a high-entropy prompt and compares the
first generated token's entropy:

- "The capital of France is" -> mass concentrated on "Paris" -> low H
- "My favorite number is"    -> spread over many numbers     -> high H

Usage:
  # vLLM:   vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-logprobs 20
  python scripts/verify_live.py --base-url http://localhost:8000/v1

  # OpenAI (no GPU needed):
  python scripts/verify_live.py --base-url https://api.openai.com/v1 \
      --api-key $OPENAI_API_KEY --model gpt-4o-mini

Only stdlib + entropy-lens are used (no openai package required).
"""

import argparse
import json
import os
import sys
import urllib.request

from entropy_lens import perplexity_from_entropy
from entropy_lens.adapters import from_openai_response

LOW_PROMPT = "Complete this sentence with one word: The capital of France is"
HIGH_PROMPT = "Complete this sentence with one number: My favorite number is"


def request_json(url: str, payload: dict | None, api_key: str) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def pick_model(base_url: str, api_key: str) -> str:
    models = request_json(f"{base_url}/models", None, api_key)
    return models["data"][0]["id"]


def chat(base_url: str, api_key: str, model: str, prompt: str, top_logprobs: int) -> dict:
    return request_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        },
        api_key,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=None, help="default: first model from /models")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument(
        "--save-json",
        default=None,
        metavar="PATH",
        help="save the low-entropy case's raw response as a fixture",
    )
    args = ap.parse_args()

    model = args.model or pick_model(args.base_url, args.api_key)
    print(f"server: {args.base_url}\nmodel:  {model}\n")

    results = {}
    for name, prompt in [("low", LOW_PROMPT), ("high", HIGH_PROMPT)]:
        response = chat(args.base_url, args.api_key, model, prompt, args.top_logprobs)
        traj = from_openai_response(response, split=None)
        results[name] = (prompt, response, traj)

    print(f"{'case':<6} | {'first token':<14} | {'H (bits)':>8} | {'perplexity':>10}")
    print("-" * 50)
    for name, (prompt, _, traj) in results.items():
        h = float(traj.entropies[0])
        print(
            f"{name:<6} | {traj.tokens[0]!r:<14} | {h:8.4f} | "
            f"{float(perplexity_from_entropy(h)):10.3f}   <- {prompt!r}"
        )

    h_low = float(results["low"][2].entropies[0])
    h_high = float(results["high"][2].entropies[0])
    print()
    if h_low < h_high:
        print(f"OK: H(low-entropy case) = {h_low:.4f} < H(high-entropy case) = {h_high:.4f}")
    else:
        print(
            f"WARNING: expected H(low) < H(high) but got {h_low:.4f} >= {h_high:.4f}. "
            "This can happen with heavily-tuned models; try another model or prompt."
        )

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results["low"][1], f, indent=2, ensure_ascii=False)
        print(f"saved raw response to {args.save_json}")

    sys.exit(0 if h_low < h_high else 1)


if __name__ == "__main__":
    main()
