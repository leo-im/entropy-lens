#!/usr/bin/env python
"""Parse a saved OpenAI-compatible response and print a per-token entropy table.

Run: python scripts/verify_adapter.py [tests/fixtures/vllm_response.json]

What to look for: obvious continuations (function words, the "Paris" after
"The capital of France is") should show low H, while positions with several
plausible continuations (sequence openers, punctuation) show higher H.
"""

import json
import sys
from pathlib import Path

from entropy_lens import perplexity_from_entropy
from entropy_lens.adapters import from_openai_response


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/vllm_response.json")
    response = json.loads(path.read_text())
    traj = from_openai_response(response)

    print(f"source: {path}")
    print(f"model:  {response.get('model', '?')}")
    print(f"text:   {traj.text!r}\n")

    print(f"{'token':<14} | {'H (bits)':>8} | {'perplexity':>10}")
    print("-" * 40)
    for tok, h in zip(traj.tokens, traj.entropies, strict=True):
        pp = perplexity_from_entropy(h)
        print(f"{tok!r:<14} | {h:8.4f} | {pp:10.3f}")

    s = traj.summary()
    print(
        f"\nsummary: mean H = {s['mean_entropy']:.4f} bits, "
        f"max = {s['max_entropy']:.4f}, min = {s['min_entropy']:.4f}, "
        f"{s['n_steps']} step(s)"
    )


if __name__ == "__main__":
    main()
