"""The interface every step that calls a language model goes through.

It is small on purpose: chat completion plus an optional read-out of the top
log-probabilities, both of which providers support in much the same shape.
Concrete implementations live in :mod:`studentsim.baselines`, and
:func:`open_client` is where one is chosen, so a different provider can be
dropped in without touching the steps that use it.

``Message.role`` is constrained to ``"system" | "user" | "assistant"``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn. ``content`` is text, or the provider's list of content parts.

    Almost every step here sends text. The chess judge sends a picture of the
    board alongside its question, and providers take that as a list of parts
    rather than a string, so the field accepts either and implementations pass
    it through untouched.
    """

    role: Role
    content: str | Sequence[Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A single completion plus optional top-k logprobs over the first generated token.

    ``top_logprobs`` is keyed by candidate string (e.g., ``"A"``, ``"B"``, ``"C"``,
    ``"D"`` for math fidelity); the value is the natural-log probability returned
    by the provider. Tokens not in the returned top-k get a floor handled by the
    caller (:data:`studentsim.domains.math.fidelity.LOGPROB_FLOOR` for math).
    """

    text: str
    top_logprobs: Mapping[str, float] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Stateless chat-completion client.

    Implementations should be safe to call from multiple threads or async tasks
    (most provider SDKs already are). They must NOT carry mutable conversation
    state; the messages list is the entire input.
    """

    name: str
    """Short identifier for logging, e.g., ``"llamacpp/qwen38-code"``."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int | None = None,
    ) -> LLMResponse:
        """Run one chat completion.

        Parameters
        ----------
        messages
            The full conversation; the first turn may be ``system``.
        max_tokens
            Token budget for the assistant's reply.
        temperature, top_p
            Sampling parameters; ``temperature=0.0`` means greedy.
        top_logprobs
            If set, request that many top-k logprobs on the first generated token.
            Used by :class:`studentsim.domains.math.fidelity.MathFidelity` for the
            four-way multiple-choice metric.
        """
        ...


def open_client(model: str) -> LLMClient:
    """Open the configured LLM provider for ``model``.

    ``STUDENTSIM_LLM_PROVIDER`` selects the backend.  The fork defaults to
    ``llamacpp`` so API-calling steps stay local unless Azure is explicitly
    requested.
    """
    provider = os.environ.get("STUDENTSIM_LLM_PROVIDER", "llamacpp").strip().lower()

    if provider in {"llamacpp", "llama.cpp", "local"}:
        from studentsim.baselines import LlamaCppClient

        base_url = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1")
        scoring_base_url = os.environ.get("LLAMACPP_SCORING_BASE_URL", base_url)
        local_model = os.environ.get("LLAMACPP_MODEL", model or "qwen38-code")
        return LlamaCppClient(
            model=local_model,
            base_url=base_url,
            scoring_base_url=scoring_base_url,
        )

    if provider in {"azure", "azure_openai", "azure-openai"}:
        from studentsim.baselines import AzureOpenAIClient

        return AzureOpenAIClient(deployment=model)

    raise ValueError(
        "Unsupported STUDENTSIM_LLM_PROVIDER "
        f"{provider!r}; expected 'llamacpp' or 'azure_openai'."
    )
