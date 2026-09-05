"""OpenAI-compatible llama.cpp :class:`LLMClient` implementation.

This backend is intended for local-first use with ``llama-server``.  It keeps
StudentSim's provider boundary identical to the Azure implementation while
allowing generation, judging, and fidelity work to stay on a local model.

Two base URLs are supported because some llama.cpp configurations use
speculative/draft decoding for throughput while log-probability-sensitive
scoring is safer on a non-speculative server profile.  When no scoring URL is
provided, both kinds of calls use the same server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from studentsim.core.llm import LLMClient, LLMResponse, Message

_DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"


class LlamaCppClient(LLMClient):
    """Chat-completion client for a local OpenAI-compatible llama.cpp server.

    Parameters
    ----------
    model
        Model alias exposed by ``llama-server`` (for example ``qwen38-code``).
    base_url
        OpenAI-compatible base URL used for ordinary generation.
    scoring_base_url
        Optional base URL used only when ``top_logprobs`` are requested.  This
        is useful when scoring runs use a llama.cpp profile with speculative
        decoding disabled.  Defaults to ``base_url``.
    api_key
        Placeholder key supplied to the OpenAI SDK.  A stock local llama.cpp
        server does not validate it.
    _inner, _scoring_inner
        Optional injected clients used by tests.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        scoring_base_url: str | None = None,
        api_key: str = "local-llamacpp",
        _inner: Any | None = None,
        _scoring_inner: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if not base_url:
            raise ValueError("base_url must be non-empty")

        self.name = f"llamacpp/{model}"
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._scoring_base_url = (scoring_base_url or base_url).rstrip("/")

        if _inner is not None:
            self._client = _inner
        else:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(base_url=self._base_url, api_key=api_key)

        if _scoring_inner is not None:
            self._scoring_client = _scoring_inner
        elif self._scoring_base_url == self._base_url:
            self._scoring_client = self._client
        else:
            from openai import OpenAI  # lazy import

            self._scoring_client = OpenAI(
                base_url=self._scoring_base_url,
                api_key=api_key,
            )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_logprobs is not None:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        client = self._scoring_client if top_logprobs is not None else self._client
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""

        top_lp_map: dict[str, float] | None = None
        if top_logprobs is not None and choice.logprobs and choice.logprobs.content:
            first_token = choice.logprobs.content[0]
            top_lp_map = {
                token.token: float(token.logprob)
                for token in first_token.top_logprobs
            }

        return LLMResponse(text=text, top_logprobs=top_lp_map)
