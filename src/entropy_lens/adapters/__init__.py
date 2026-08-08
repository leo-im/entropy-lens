"""Adapters that turn provider-specific outputs into EntropyTrajectory."""

from entropy_lens.adapters.openai_compat import from_openai_response

__all__ = ["from_openai_response"]
