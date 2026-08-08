import json
import math
from pathlib import Path

import numpy as np
import pytest

from entropy_lens.adapters import from_openai_response
from entropy_lens.adapters.hf_transformers import from_hf_generate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def vllm_response() -> dict:
    return json.loads((FIXTURES / "vllm_response.json").read_text())


class TestChatFormat:
    def test_basic_parse(self, vllm_response):
        traj = from_openai_response(vllm_response)
        assert traj.text == "The capital of France is Paris."
        assert len(traj.tokens) == 7
        assert traj.entropies.shape == (7,)
        assert np.all(traj.entropies >= 0)

    def test_confident_token_has_low_entropy(self, vllm_response):
        traj = from_openai_response(vllm_response)
        h = dict(zip(traj.tokens, traj.entropies, strict=True))
        # " Paris" (p~0.99) must be far more certain than the opener "The".
        assert h[" Paris"] < 0.1
        assert h["The"] > 1.0
        assert h[" Paris"] < h["The"]

    def test_default_split_is_sentence(self, vllm_response):
        traj = from_openai_response(vllm_response)
        assert traj.step_boundaries == [0]  # single sentence -> single step

    def test_split_none(self, vllm_response):
        traj = from_openai_response(vllm_response, split=None)
        assert traj.step_boundaries == [0]

    def test_custom_pattern(self, vllm_response):
        traj = from_openai_response(vllm_response, pattern=r" is")
        assert traj.n_steps == 2

    def test_base_nats(self, vllm_response):
        bits = from_openai_response(vllm_response).entropies
        nats = from_openai_response(vllm_response, base="nats").entropies
        np.testing.assert_allclose(bits, nats / np.log(2))

    def test_tail_uniform(self, vllm_response):
        ig = from_openai_response(vllm_response).entropies
        un = from_openai_response(vllm_response, tail="uniform", vocab_size=151_936).entropies
        assert np.all(un >= ig)

    def test_model_dump_object(self, vllm_response):
        class FakeSDKResponse:
            def model_dump(self):
                return vllm_response

        traj = from_openai_response(FakeSDKResponse())
        assert len(traj.tokens) == 7

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="model_dump"):
            from_openai_response(["not", "a", "response"])

    def test_no_choices(self):
        with pytest.raises(ValueError, match="choices"):
            from_openai_response({"object": "chat.completion"})

    def test_choice_index_out_of_range(self, vllm_response):
        with pytest.raises(IndexError, match="choice_index"):
            from_openai_response(vllm_response, choice_index=3)

    def test_missing_logprobs(self):
        resp = {"choices": [{"message": {"content": "hi"}, "logprobs": None}]}
        with pytest.raises(ValueError, match="logprobs=True"):
            from_openai_response(resp)

    def test_unrecognized_logprobs_shape(self):
        resp = {"choices": [{"logprobs": {"weird": []}}]}
        with pytest.raises(ValueError, match="unrecognized"):
            from_openai_response(resp)

    def test_empty_top_logprobs_falls_back_to_own_logprob(self):
        resp = {
            "choices": [
                {"logprobs": {"content": [{"token": "a", "logprob": -0.5, "top_logprobs": []}]}}
            ]
        }
        traj = from_openai_response(resp)
        # Single known candidate -> degenerate estimate of 0.
        assert traj.entropies[0] == pytest.approx(0.0)


class TestLlamaCppCapture:
    """Real response captured from llama.cpp serving Qwen2.5-0.5B-Instruct."""

    @pytest.fixture
    def response(self) -> dict:
        return json.loads((FIXTURES / "llamacpp_response.json").read_text())

    def test_parse(self, response):
        traj = from_openai_response(response)
        assert traj.text == "Paris."
        assert traj.entropies.shape == (len(traj.tokens),)
        assert np.all(traj.entropies >= 0)

    def test_top_k_present(self, response):
        content = response["choices"][0]["logprobs"]["content"]
        assert all(len(e["top_logprobs"]) == 20 for e in content)

    def test_first_token_is_confident(self, response):
        # "The capital of France is" -> "Paris" should be low entropy.
        traj = from_openai_response(response)
        assert traj.tokens[0] == "Paris"
        assert traj.entropies[0] < 1.0


class TestLegacyFormat:
    def make_response(self) -> dict:
        return {
            "choices": [
                {
                    "text": " Paris.",
                    "logprobs": {
                        "tokens": [" Paris", "."],
                        "token_logprobs": [math.log(0.9), math.log(0.7)],
                        "top_logprobs": [
                            {" Paris": math.log(0.9), " Lyon": math.log(0.05)},
                            {".": math.log(0.7), ",": math.log(0.2)},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    def test_parse(self):
        traj = from_openai_response(self.make_response())
        assert traj.tokens == [" Paris", "."]
        assert traj.entropies.shape == (2,)
        assert traj.entropies[0] < traj.entropies[1]  # 0.9/0.05 vs 0.7/0.2

    def test_missing_top_logprobs_uses_token_logprobs(self):
        resp = self.make_response()
        resp["choices"][0]["logprobs"]["top_logprobs"] = None
        traj = from_openai_response(resp)
        np.testing.assert_allclose(traj.entropies, [0.0, 0.0], atol=1e-12)


class FakeTokenizer:
    """Maps token id -> single character (id 0 -> 'a', 1 -> 'b', ...)."""

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


class FakeGenerateOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class TestHFAdapter:
    def test_full_vocab_entropy(self):
        # 2 generated tokens over a 4-token vocab: uniform then one-hot-ish.
        uniform = np.zeros((1, 4))
        peaked = np.array([[100.0, 0.0, 0.0, 0.0]])
        out = FakeGenerateOutput(
            sequences=np.array([[9, 9, 0, 1]]),  # prompt ids 9,9 then gen 0,1
            scores=[uniform, peaked],
        )
        traj = from_hf_generate(out, FakeTokenizer(), split=None)
        assert traj.tokens == ["a", "b"]
        np.testing.assert_allclose(traj.entropies, [2.0, 0.0], atol=1e-9)

    def test_masked_vocab_entries(self):
        # -inf logits (suppressed tokens) must contribute zero, not NaN.
        scores = [np.array([[0.0, 0.0, -np.inf, -np.inf]])]
        out = FakeGenerateOutput(sequences=np.array([[5, 0]]), scores=scores)
        traj = from_hf_generate(out, FakeTokenizer(), split=None)
        assert traj.entropies[0] == pytest.approx(1.0)

    def test_nats(self):
        scores = [np.zeros((1, 4))]
        out = FakeGenerateOutput(sequences=np.array([[5, 0]]), scores=scores)
        traj = from_hf_generate(out, FakeTokenizer(), base="nats", split=None)
        assert traj.entropies[0] == pytest.approx(np.log(4))

    def test_missing_scores(self):
        with pytest.raises(ValueError, match="output_scores"):
            from_hf_generate(object(), FakeTokenizer())

    def test_empty_scores(self):
        out = FakeGenerateOutput(sequences=np.array([[1]]), scores=[])
        with pytest.raises(ValueError, match="empty"):
            from_hf_generate(out, FakeTokenizer())

    def test_invalid_base(self):
        out = FakeGenerateOutput(sequences=np.array([[1, 0]]), scores=[np.zeros((1, 2))])
        with pytest.raises(ValueError, match="base"):
            from_hf_generate(out, FakeTokenizer(), base="hartleys")
