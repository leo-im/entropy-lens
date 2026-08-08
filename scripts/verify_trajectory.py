#!/usr/bin/env python
"""CoT trajectory verification against a live OpenAI-compatible server.

Asks a GSM8K-style arithmetic question with a CoT prompt, splits the answer
into sentence steps, prints per-step statistics and saves the trajectory plot.

Usage:
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-logprobs 20   # terminal 1
  python scripts/verify_trajectory.py --base-url http://localhost:8000/v1

Things to eyeball in the output:
- step boundaries should fall on sentence ends;
- the final-answer step typically (not always) shows lower mean H.
"""

import argparse
import os
from pathlib import Path

from verify_live import chat, pick_model

from entropy_lens.adapters import from_openai_response
from entropy_lens.viz import plot_trajectory

PROBLEM = (
    "A shop sells pencils in boxes of 12. Maria buys 3 boxes and gives away "
    "7 pencils. How many pencils does she have left? "
    "Think step by step in short sentences, then end with 'The answer is N.'"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--out", default="verify_output/trajectory.png")
    args = ap.parse_args()

    model = args.model or pick_model(args.base_url, args.api_key)
    print(f"server: {args.base_url}\nmodel:  {model}\n")

    response = chat(args.base_url, args.api_key, model, PROBLEM, args.top_logprobs, max_tokens=256)
    traj = from_openai_response(response, split="sentence")

    print(f"completion ({len(traj.tokens)} tokens, {traj.n_steps} steps):\n")
    means = traj.step_means()
    for i, (text, mean) in enumerate(zip(traj.step_texts(), means, strict=True), 1):
        shown = text.strip().replace("\n", " ")
        if len(shown) > 70:
            shown = shown[:67] + "..."
        print(f"  step {i}: mean H = {mean:.4f} bits | {shown}")

    s = traj.summary()
    print(
        f"\nsummary: total decrease = {s['total_decrease']:.4f} bits "
        f"(first step {s['first_step_mean']:.4f} -> last step {s['last_step_mean']:.4f}), "
        f"monotonic fraction = {s['monotonic_fraction']:.2f}"
    )
    if means[-1] <= means.mean():
        print("final step is at or below the average H — typical convergence pattern")
    else:
        print("note: final step H is above average (not unusual, but worth a look)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_trajectory(traj, title=f"CoT entropy trajectory — {model}")
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot to {out}")


if __name__ == "__main__":
    main()
