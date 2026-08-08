#!/usr/bin/env python
"""Mathematical sanity checks for entropy-lens core (no LLM required).

Run: python scripts/verify_math.py
Every check both asserts and prints a human-readable line.
"""

import numpy as np

from entropy_lens import perplexity_from_entropy, token_entropy, token_varentropy


def check(label: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    assert ok, f"{label} failed: {detail}"


def main() -> None:
    print("=== entropy-lens math sanity checks ===\n")

    # 1. Uniform k=4 -> H = 2 bits, perplexity = 4.
    lp = np.full(4, np.log(0.25))
    h = token_entropy(lp)
    pp = perplexity_from_entropy(h)
    check(
        "uniform k=4",
        abs(h - 2.0) < 1e-12 and abs(pp - 4.0) < 1e-9,
        f"H = {h:.6f} bits (expected 2.0), perplexity = {pp:.6f} (expected 4.0)",
    )

    # 2. One-hot -> H ~ 0, perplexity ~ 1.
    eps = 1e-12
    p = np.array([1.0 - 3 * eps, eps, eps, eps])
    h = token_entropy(np.log(p))
    pp = perplexity_from_entropy(h)
    check(
        "one-hot",
        h < 1e-9 and abs(pp - 1.0) < 1e-6,
        f"H = {h:.3e} bits (~0), perplexity = {pp:.9f} (~1)",
    )

    # 3. Sharpening (temperature down) -> H monotonically decreases.
    logits = np.array([2.0, 1.0, 0.5, -1.0, -2.0])
    temps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    entropies = []
    for t in temps:
        scaled = logits / t
        lp = scaled - np.log(np.sum(np.exp(scaled)))
        entropies.append(token_entropy(lp))
    monotone = all(a > b for a, b in zip(entropies, entropies[1:], strict=False))
    table = ", ".join(f"T={t}: {h:.4f}" for t, h in zip(temps, entropies, strict=False))
    check("sharpening monotone", monotone, f"H(bits) as T drops -> {table}")

    # 4. nats <-> bits: H_bits = H_nats / ln 2.
    lp = np.log(np.array([0.6, 0.25, 0.1, 0.05]))
    h_bits = token_entropy(lp, base="bits")
    h_nats = token_entropy(lp, base="nats")
    check(
        "nats/bits conversion",
        abs(h_bits - h_nats / np.log(2)) < 1e-12,
        f"H = {h_nats:.6f} nats = {h_bits:.6f} bits (H_nats/ln2 = {h_nats / np.log(2):.6f})",
    )

    # 5. Tail handling: ignore (renormalize) vs uniform spread.
    lp = np.log(np.array([0.5, 0.2, 0.1]))  # 0.2 residual mass
    h_ig = token_entropy(lp, tail="ignore")
    h_un = token_entropy(lp, tail="uniform", vocab_size=32000)
    check(
        "tail bounds",
        h_ig < h_un,
        f'top-3 with 0.2 residual: tail="ignore" H = {h_ig:.4f} bits '
        f'< tail="uniform" H = {h_un:.4f} bits (true H lies in between)',
    )

    # 6. Varentropy: uniform -> 0 despite max H; tiered -> large V.
    v_uniform = token_varentropy(np.full(8, np.log(1 / 8)))
    p_tiered = np.array([0.75, 0.125, 0.125])
    v_tiered = token_varentropy(np.log(p_tiered))
    s = -np.log2(p_tiered)
    h = float(np.sum(p_tiered * s))
    v_expected = float(np.sum(p_tiered * (s - h) ** 2))
    check(
        "varentropy shape",
        v_uniform < 1e-12 and abs(v_tiered - v_expected) < 1e-9,
        f"uniform k=8: V = {v_uniform:.2e} (max H but constant surprisal -> 0); "
        f"tiered (0.75,0.125,0.125): V = {v_tiered:.4f} bit^2 "
        f"(analytic {v_expected:.4f})",
    )

    print("\nAll math sanity checks passed.")


if __name__ == "__main__":
    main()
