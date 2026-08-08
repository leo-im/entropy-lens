#!/usr/bin/env python
"""Statistical low-vs-high entropy contrast battery against a live server.

verify_live.py checks a single prompt pair (a smoke test); this script runs a
battery of pairs and reports aggregate statistics, because any individual pair
can flip depending on the model. Each pair contrasts a prompt with an
(almost) forced continuation against one with many plausible continuations,
comparing the entropy of the first generated token.

Pass criterion: H(low) < H(high) in at least --min-win-rate of pairs
(default 0.8).

Usage:
  llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --port 8000
  python scripts/verify_battery.py --base-url http://127.0.0.1:8000/v1
"""

import argparse
import json
import os
import sys

from verify_live import chat, pick_model

from entropy_lens.adapters import from_openai_response

# (label, low-entropy prompt, high-entropy prompt)
#
# Design notes: both prompts of a pair share the same output-format
# instruction, so the first token's entropy contrasts *semantic* uncertainty
# (one canonical answer vs. many valid answers) rather than formatting
# choices (capitalization, preamble phrasing), which otherwise dominate the
# first-token distribution.
WORD = " Reply with exactly one lowercase word."
DIGIT = " Reply with just the digits."
LETTER = " Reply with one lowercase letter."
PAIRS = [
    (
        "capital-fr/any-city",
        "What is the capital of France?" + WORD,
        "Name any city in the world." + WORD,
    ),
    (
        "2+2/random-digit",
        "What is 2 + 2?" + DIGIT,
        "Pick a random number between 1 and 9." + DIGIT,
    ),
    (
        "antonym-hot/any-animal",
        "What is the opposite of hot?" + WORD,
        "Name any animal." + WORD,
    ),
    (
        "sky-color/any-color",
        "What color is a clear daytime sky?" + WORD,
        "Name any color." + WORD,
    ),
    (
        "symbol-au/any-metal",
        "Which metal does the chemical symbol Au stand for?" + WORD,
        "Name any metal." + WORD,
    ),
    (
        "days-in-week/any-day",
        "How many days are in a week?" + DIGIT,
        "Pick a random number between 1 and 31." + DIGIT,
    ),
    (
        "first-letter/any-letter",
        "What is the first letter of the English alphabet?" + LETTER,
        "Pick any letter of the alphabet." + LETTER,
    ),
    (
        "sunrise/any-direction",
        "In which compass direction does the sun rise?" + WORD,
        "Pick any compass direction." + WORD,
    ),
    (
        "freezing-c/any-temp",
        "At what Celsius temperature does water freeze at sea level?" + DIGIT,
        "Pick a random Celsius temperature between 0 and 99." + DIGIT,
    ),
    (
        "capital-jp/any-asian-city",
        "What is the capital of Japan?" + WORD,
        "Name any city in Asia." + WORD,
    ),
    (
        "antonym-up/any-word",
        "What is the opposite of up?" + WORD,
        "Say any English word." + WORD,
    ),
    (
        "after-monday/any-day",
        "Which day of the week comes right after Monday?" + WORD,
        "Pick any day of the week." + WORD,
    ),
]


def first_token_entropy(base_url, api_key, model, prompt, top_logprobs) -> tuple[str, float]:
    response = chat(base_url, api_key, model, prompt, top_logprobs)
    traj = from_openai_response(response, split=None)
    return traj.tokens[0], float(traj.entropies[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--min-win-rate", type=float, default=0.8)
    ap.add_argument("--json-out", default=None, metavar="PATH")
    args = ap.parse_args()

    model = args.model or pick_model(args.base_url, args.api_key)
    print(f"server: {args.base_url}\nmodel:  {model}\n")
    print(f"{'pair':<22} | {'H(low)':>7} | {'H(high)':>7} | {'diff':>7} | result")
    print("-" * 66)

    results = []
    for label, low_prompt, high_prompt in PAIRS:
        tok_low, h_low = first_token_entropy(
            args.base_url, args.api_key, model, low_prompt, args.top_logprobs
        )
        tok_high, h_high = first_token_entropy(
            args.base_url, args.api_key, model, high_prompt, args.top_logprobs
        )
        win = h_low < h_high
        results.append(
            {
                "pair": label,
                "h_low": h_low,
                "h_high": h_high,
                "token_low": tok_low,
                "token_high": tok_high,
                "win": win,
            }
        )
        mark = "ok" if win else "FLIP"
        print(f"{label:<22} | {h_low:7.4f} | {h_high:7.4f} | {h_high - h_low:+7.4f} | {mark}")

    wins = sum(r["win"] for r in results)
    n = len(results)
    win_rate = wins / n
    mean_diff = sum(r["h_high"] - r["h_low"] for r in results) / n
    print("-" * 66)
    print(f"win rate: {wins}/{n} = {win_rate:.0%}   mean H(high) - H(low) = {mean_diff:+.4f} bits")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"model": model, "win_rate": win_rate, "results": results}, f, indent=2)
        print(f"saved results to {args.json_out}")

    if win_rate >= args.min_win_rate:
        print(f"PASS: win rate {win_rate:.0%} >= threshold {args.min_win_rate:.0%}")
        sys.exit(0)
    print(f"FAIL: win rate {win_rate:.0%} < threshold {args.min_win_rate:.0%}")
    sys.exit(1)


if __name__ == "__main__":
    main()
