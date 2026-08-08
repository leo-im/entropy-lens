import numpy as np
import pytest

from entropy_lens.trajectory import EntropyTrajectory, split_steps


def tokenize(text: str) -> list[str]:
    """Crude whitespace-preserving tokenizer for tests (whitespace attaches left)."""
    tokens = []
    word = ""
    for ch in text:
        if ch in " \n":
            word += ch
        else:
            if word.endswith((" ", "\n")):
                tokens.append(word)
                word = ""
            word += ch
    if word:
        tokens.append(word)
    return tokens


class TestSplitSteps:
    def test_sentence_split(self):
        tokens = ["First", " sentence", ".", " Second", " one", ".", " Third", "."]
        bounds = split_steps(tokens, "sentence")
        assert bounds == [0, 3, 6]

    def test_single_sentence(self):
        tokens = ["Only", " one", " sentence", "."]
        assert split_steps(tokens, "sentence") == [0]

    def test_trailing_punctuation_no_extra_step(self):
        # Final "." should not open an empty step.
        tokens = ["Done", "."]
        assert split_steps(tokens, "sentence") == [0]

    def test_question_and_exclamation(self):
        tokens = ["Really", "?", " Yes", "!", " Ok", "."]
        assert split_steps(tokens, "sentence") == [0, 2, 4]

    def test_paragraph_split(self):
        tokens = ["Para one", ".", "\n\n", "Para two", "."]
        assert split_steps(tokens, "paragraph") == [0, 3]

    def test_step_marker_split(self):
        text = "Step 1: add numbers.\nStep 2: multiply.\nStep 3: done."
        tokens = tokenize(text)
        bounds = split_steps(tokens, "step_marker")
        assert len(bounds) == 3
        assert bounds[0] == 0
        # Each boundary token should begin a "Step N" marker.
        for b in bounds[1:]:
            assert "Step" in tokens[b]

    def test_custom_pattern(self):
        tokens = ["a", "|", "b", "|", "c"]
        bounds = split_steps(tokens, pattern=r"\|", boundary="end")
        assert bounds == [0, 2, 4]

    def test_custom_pattern_start_boundary(self):
        tokens = ["intro ", "- item one ", "- item two"]
        bounds = split_steps(tokens, pattern=r"- ", boundary="start")
        assert bounds == [0, 1, 2]

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            split_steps(["a"], "chapter")

    def test_invalid_boundary(self):
        with pytest.raises(ValueError, match="boundary"):
            split_steps(["a"], pattern=r"x", boundary="middle")

    def test_empty_tokens(self):
        assert split_steps([]) == []

    def test_period_inside_number_not_a_boundary(self):
        tokens = ["Pi", " is", " 3", ".", "14", " roughly", "."]
        # "3.14" has no whitespace after the dot, so no split there.
        assert split_steps(tokens, "sentence") == [0]


class TestSentenceSplitRealWorld:
    """Patterns that break naive sentence splitters in real CoT text."""

    def check(self, text: str, expected_step_starts: list[str]):
        tokens = tokenize(text)
        bounds = split_steps(tokens, "sentence")
        got = [tokens[b].strip() for b in bounds]
        assert got == expected_step_starts, f"boundaries {bounds} -> {got}"

    def test_title_abbreviation(self):
        self.check("Dr. Smith arrived early. He sat down.", ["Dr.", "He"])

    def test_multiple_titles(self):
        self.check("Mr. and Mrs. Lee left. Prof. Kim stayed.", ["Mr.", "Prof."])

    def test_eg_abbreviation(self):
        self.check("Add fruit, e.g. apples or pears. Then mix well.", ["Add", "Then"])

    def test_ie_abbreviation(self):
        self.check("Use the base, i.e. bits. Then convert.", ["Use", "Then"])

    def test_initials(self):
        self.check("I met J. Smith yesterday. We talked.", ["I", "We"])

    def test_dotted_acronym(self):
        self.check("She moved to the U.S. in May. Life changed.", ["She", "Life"])

    def test_decimal_then_sentence(self):
        self.check("The rate is 3.14 percent. That is high.", ["The", "That"])

    def test_ellipsis_and_newline(self):
        self.check("Hmm... let me think.\nThe answer is 4.", ["Hmm...", "The"])

    def test_numbered_list_lines(self):
        text = "1. Add the numbers. 2. Multiply by two."
        tokens = tokenize(text)
        bounds = split_steps(tokens, "sentence")
        # "1." / "2." are list markers; we accept splits at sentence ends only.
        assert [tokens[b].strip() for b in bounds] == ["1.", "2."]

    def test_abbreviation_at_text_start(self):
        self.check("Dr. Lee spoke. Everyone listened.", ["Dr.", "Everyone"])

    def test_lowercase_word_ending_sentence_still_splits(self):
        # "no" is not in the abbreviation list on purpose.
        self.check("I said no. Then I left.", ["I", "Then"])


def make_trajectory() -> EntropyTrajectory:
    # 6 tokens, 3 steps of 2 tokens with means 3.0, 2.0, 1.0.
    return EntropyTrajectory(
        entropies=np.array([3.5, 2.5, 2.0, 2.0, 1.5, 0.5]),
        tokens=["a", "b", "c", "d", "e", "f"],
        step_boundaries=[0, 2, 4],
    )


class TestEntropyTrajectory:
    def test_step_means(self):
        np.testing.assert_allclose(make_trajectory().step_means(), [3.0, 2.0, 1.0])

    def test_delta_h(self):
        np.testing.assert_allclose(make_trajectory().delta_h(), [-1.0, -1.0])

    def test_summary(self):
        s = make_trajectory().summary()
        assert s["n_tokens"] == 6
        assert s["n_steps"] == 3
        assert s["mean_entropy"] == pytest.approx(2.0)
        assert s["max_entropy"] == 3.5
        assert s["min_entropy"] == 0.5
        assert s["total_decrease"] == pytest.approx(2.0)
        assert s["monotonic_fraction"] == 1.0

    def test_summary_single_step(self):
        traj = EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"])
        s = traj.summary()
        assert s["n_steps"] == 1
        assert np.isnan(s["monotonic_fraction"])
        assert s["total_decrease"] == 0.0

    def test_default_single_step(self):
        traj = EntropyTrajectory(np.array([1.0]), ["x"])
        assert traj.step_boundaries == [0]
        assert traj.n_steps == 1

    def test_step_texts_and_text(self):
        traj = make_trajectory()
        assert traj.text == "abcdef"
        assert traj.step_texts() == ["ab", "cd", "ef"]

    def test_delta_h_empty_for_single_step(self):
        traj = EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"])
        assert traj.delta_h().size == 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            EntropyTrajectory(np.array([1.0]), ["a", "b"])

    def test_bad_boundaries_not_starting_at_zero(self):
        with pytest.raises(ValueError, match="start at 0"):
            EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"], [1])

    def test_bad_boundaries_not_increasing(self):
        with pytest.raises(ValueError, match="increasing"):
            EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"], [0, 0])

    def test_boundary_out_of_range(self):
        with pytest.raises(ValueError, match="beyond"):
            EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"], [0, 2])

    def test_non_1d_entropies(self):
        with pytest.raises(ValueError, match="1-D"):
            EntropyTrajectory(np.ones((2, 2)), ["a", "b"])

    def test_varentropies_optional(self):
        traj = make_trajectory()
        assert traj.varentropies is None
        s = traj.summary()
        assert s["mean_varentropy"] is None and s["max_varentropy"] is None
        with pytest.raises(ValueError, match="no varentropies"):
            traj.step_varentropy_means()

    def test_varentropies_attached(self):
        traj = EntropyTrajectory(
            np.array([1.0, 2.0, 3.0, 4.0]),
            ["a", "b", "c", "d"],
            [0, 2],
            varentropies=np.array([0.1, 0.3, 0.5, 0.7]),
        )
        np.testing.assert_allclose(traj.step_varentropy_means(), [0.2, 0.6])
        s = traj.summary()
        assert s["mean_varentropy"] == pytest.approx(0.4)
        assert s["max_varentropy"] == pytest.approx(0.7)

    def test_varentropies_length_mismatch(self):
        with pytest.raises(ValueError, match="varentropies"):
            EntropyTrajectory(np.array([1.0, 2.0]), ["a", "b"], varentropies=np.array([0.1]))
