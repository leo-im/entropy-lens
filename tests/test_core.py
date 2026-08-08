import numpy as np
import pytest

from entropy_lens.core import (
    convert_entropy,
    perplexity_from_entropy,
    sequence_entropies,
    sequence_varentropies,
    token_entropy,
    token_varentropy,
)


def uniform_logprobs(k: int) -> np.ndarray:
    return np.full(k, np.log(1.0 / k))


class TestTokenEntropy:
    def test_uniform_k4_is_2_bits(self):
        assert token_entropy(uniform_logprobs(4)) == pytest.approx(2.0)

    @pytest.mark.parametrize("k", [2, 8, 16, 100])
    def test_uniform_k_is_log2_k(self, k):
        assert token_entropy(uniform_logprobs(k)) == pytest.approx(np.log2(k))

    def test_one_hot_is_zero(self):
        lp = np.log(np.array([1.0 - 1e-12, 1e-12 / 3, 1e-12 / 3, 1e-12 / 3]))
        assert token_entropy(lp) == pytest.approx(0.0, abs=1e-9)

    def test_sharpening_decreases_entropy(self):
        # Lowering temperature (sharpening the softmax) must reduce H.
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        entropies = []
        for temp in [2.0, 1.0, 0.5, 0.1]:
            scaled = logits / temp
            lp = scaled - np.log(np.sum(np.exp(scaled)))
            entropies.append(token_entropy(lp))
        assert all(a > b for a, b in zip(entropies, entropies[1:], strict=False))

    def test_nats_bits_conversion(self):
        lp = np.log(np.array([0.5, 0.3, 0.2]))
        h_bits = token_entropy(lp, base="bits")
        h_nats = token_entropy(lp, base="nats")
        assert h_bits == pytest.approx(h_nats / np.log(2))

    def test_ignore_renormalizes_partial_mass(self):
        # Top-2 of some larger distribution: [0.3, 0.3] renormalizes to
        # uniform over 2 -> exactly 1 bit.
        lp = np.log(np.array([0.3, 0.3]))
        assert token_entropy(lp, tail="ignore") == pytest.approx(1.0)

    def test_uniform_tail_exceeds_ignore(self):
        # Spreading residual mass over the tail adds uncertainty.
        lp = np.log(np.array([0.5, 0.2, 0.1]))  # 0.2 residual
        h_ignore = token_entropy(lp, tail="ignore")
        h_uniform = token_entropy(lp, tail="uniform", vocab_size=32000)
        assert h_uniform > h_ignore

    def test_uniform_tail_exact_value(self):
        p = np.array([0.5, 0.25])  # residual 0.25 over vocab 4 -> 2 tail tokens
        expected_nats = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 0.25 * np.log(0.25 / 2))
        h = token_entropy(np.log(p), base="nats", tail="uniform", vocab_size=4)
        assert h == pytest.approx(expected_nats)

    def test_uniform_tail_with_full_mass_falls_back(self):
        # No residual mass to spread: uniform == ignore.
        lp = uniform_logprobs(4)
        h_u = token_entropy(lp, tail="uniform", vocab_size=32000)
        assert h_u == pytest.approx(2.0)

    def test_uniform_requires_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size"):
            token_entropy(uniform_logprobs(4), tail="uniform")

    def test_uniform_rejects_small_vocab(self):
        with pytest.raises(ValueError, match="vocab_size"):
            token_entropy(uniform_logprobs(4), tail="uniform", vocab_size=4)

    def test_invalid_base(self):
        with pytest.raises(ValueError, match="base"):
            token_entropy(uniform_logprobs(4), base="hartleys")

    def test_invalid_tail(self):
        with pytest.raises(ValueError, match="tail"):
            token_entropy(uniform_logprobs(4), tail="drop")

    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            token_entropy(np.array([]))

    def test_nan_input(self):
        with pytest.raises(ValueError, match="NaN"):
            token_entropy(np.array([np.nan, -1.0]))

    def test_single_candidate(self):
        assert token_entropy(np.array([0.0])) == pytest.approx(0.0)

    def test_accepts_list_input(self):
        assert token_entropy([np.log(0.5), np.log(0.5)]) == pytest.approx(1.0)


class TestSequenceEntropies:
    def test_shape_and_values(self):
        seq = [uniform_logprobs(2), uniform_logprobs(4), uniform_logprobs(8)]
        h = sequence_entropies(seq)
        assert h.shape == (3,)
        np.testing.assert_allclose(h, [1.0, 2.0, 3.0])

    def test_variable_k(self):
        seq = [uniform_logprobs(4), np.array([np.log(1.0)])]
        h = sequence_entropies(seq)
        np.testing.assert_allclose(h, [2.0, 0.0], atol=1e-12)

    def test_empty_sequence(self):
        assert sequence_entropies([]).shape == (0,)

    def test_kwargs_forwarded(self):
        seq = [np.log(np.array([0.5, 0.2]))]
        h = sequence_entropies(seq, tail="uniform", vocab_size=100)
        assert h[0] > sequence_entropies(seq)[0]


class TestTokenVarentropy:
    @pytest.mark.parametrize("k", [2, 4, 8, 100])
    def test_uniform_is_zero(self, k):
        # Every candidate is equally surprising -> surprisal is constant.
        assert token_varentropy(uniform_logprobs(k)) == pytest.approx(0.0, abs=1e-12)

    def test_one_hot_is_zero(self):
        lp = np.log(np.array([1.0 - 1e-12, 1e-12 / 3, 1e-12 / 3, 1e-12 / 3]))
        assert token_varentropy(lp) == pytest.approx(0.0, abs=1e-6)

    def test_tiered_exact_value(self):
        # (0.75, 0.125, 0.125): surprisals differ -> V > 0, analytic value.
        p = np.array([0.75, 0.125, 0.125])
        s = -np.log2(p)
        h = float(np.sum(p * s))
        expected = float(np.sum(p * (s - h) ** 2))
        assert token_varentropy(np.log(p)) == pytest.approx(expected)
        assert expected > 1.0  # clearly tiered

    def test_tiered_beats_uniform_despite_lower_entropy(self):
        # Uniform has max H but zero V; the tiered distribution wins on V.
        tiered = np.log(np.array([0.75, 0.125, 0.125]))
        uniform = uniform_logprobs(8)
        assert token_entropy(uniform) > token_entropy(tiered)
        assert token_varentropy(uniform) < token_varentropy(tiered)

    def test_nats_bits_conversion(self):
        lp = np.log(np.array([0.6, 0.25, 0.1, 0.05]))
        v_bits = token_varentropy(lp, base="bits")
        v_nats = token_varentropy(lp, base="nats")
        assert v_bits == pytest.approx(v_nats / np.log(2) ** 2)

    def test_uniform_tail_exact_value(self):
        # (0.5, 0.25) with 0.25 residual over vocab 4 -> 2 tail tokens @ 0.125.
        p_full = np.array([0.5, 0.25, 0.125, 0.125])
        s = -np.log(p_full)
        h = float(np.sum(p_full * s))
        expected = float(np.sum(p_full * (s - h) ** 2))
        v = token_varentropy(
            np.log(np.array([0.5, 0.25])), base="nats", tail="uniform", vocab_size=4
        )
        assert v == pytest.approx(expected)

    def test_renormalization(self):
        # Half-scale top-k must give the same V as the normalized version.
        p = np.array([0.6, 0.3, 0.1])
        assert token_varentropy(np.log(p / 2)) == pytest.approx(token_varentropy(np.log(p)))

    def test_uniform_requires_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size"):
            token_varentropy(uniform_logprobs(4), tail="uniform")

    def test_invalid_base(self):
        with pytest.raises(ValueError, match="base"):
            token_varentropy(uniform_logprobs(4), base="hartleys")

    def test_invalid_tail(self):
        with pytest.raises(ValueError, match="tail"):
            token_varentropy(uniform_logprobs(4), tail="drop")

    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            token_varentropy(np.array([]))

    def test_sequence_varentropies(self):
        seq = [uniform_logprobs(4), np.log(np.array([0.75, 0.125, 0.125]))]
        v = sequence_varentropies(seq)
        assert v.shape == (2,)
        assert v[0] == pytest.approx(0.0, abs=1e-12)
        assert v[1] > 1.0


class TestPerplexity:
    def test_uniform_k4_perplexity_is_4(self):
        h = token_entropy(uniform_logprobs(4))
        assert perplexity_from_entropy(h) == pytest.approx(4.0)

    def test_zero_entropy_perplexity_is_1(self):
        assert perplexity_from_entropy(0.0) == pytest.approx(1.0)

    def test_nats_base(self):
        assert perplexity_from_entropy(np.log(4.0), base="nats") == pytest.approx(4.0)

    def test_array_input(self):
        pp = perplexity_from_entropy(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(pp, [1.0, 2.0, 4.0])

    def test_invalid_base(self):
        with pytest.raises(ValueError, match="base"):
            perplexity_from_entropy(1.0, base="dits")


class TestConvertEntropy:
    def test_roundtrip(self):
        h = 2.5
        nats = convert_entropy(h, from_base="bits", to_base="nats")
        assert nats == pytest.approx(2.5 * np.log(2))
        back = convert_entropy(nats, from_base="nats", to_base="bits")
        assert back == pytest.approx(h)

    def test_identity(self):
        assert convert_entropy(1.5, from_base="bits", to_base="bits") == 1.5
