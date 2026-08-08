"""entropy-lens: logprobs in, entropy trajectory out."""

from entropy_lens.core import (
    convert_entropy,
    perplexity_from_entropy,
    sequence_entropies,
    sequence_varentropies,
    token_entropy,
    token_varentropy,
)
from entropy_lens.trajectory import EntropyTrajectory, split_steps

__version__ = "0.2.0"

__all__ = [
    "EntropyTrajectory",
    "convert_entropy",
    "perplexity_from_entropy",
    "sequence_entropies",
    "sequence_varentropies",
    "split_steps",
    "token_entropy",
    "token_varentropy",
]
