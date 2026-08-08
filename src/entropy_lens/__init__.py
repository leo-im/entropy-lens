"""entropy-lens: logprobs in, entropy trajectory out."""

from entropy_lens.core import (
    convert_entropy,
    perplexity_from_entropy,
    sequence_entropies,
    token_entropy,
)
from entropy_lens.trajectory import EntropyTrajectory, split_steps

__version__ = "0.1.0"

__all__ = [
    "EntropyTrajectory",
    "convert_entropy",
    "perplexity_from_entropy",
    "sequence_entropies",
    "split_steps",
    "token_entropy",
]
