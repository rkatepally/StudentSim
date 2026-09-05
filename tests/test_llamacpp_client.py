from types import SimpleNamespace

from studentsim.baselines.llamacpp import LlamaCppClient
from studentsim.core.llm import Message


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def response(text="OK", top_logprobs=None):
    logprobs = None
    if top_logprobs is not None:
        tokens = [SimpleNamespace(token=t, logprob=lp) for t, lp in top_logprobs.items()]
        first = SimpleNamespace(top_logprobs=tokens)
        logprobs = SimpleNamespace(content=[first])
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        logprobs=logprobs,
    )
    return SimpleNamespace(choices=[choice])


def test_generation_uses_primary_client():
    primary = FakeClient(response("LOCAL_OK"))
    scoring = FakeClient(response("SCORE"))
    client = LlamaCppClient(
        model="qwen38-code",
        _inner=primary,
        _scoring_inner=scoring,
    )

    result = client.complete([Message("user", "hello")], max_tokens=20)

    assert result.text == "LOCAL_OK"
    assert len(primary.chat.completions.calls) == 1
    assert scoring.chat.completions.calls == []
    assert primary.chat.completions.calls[0]["model"] == "qwen38-code"


def test_logprob_request_uses_scoring_client_and_maps_first_token():
    primary = FakeClient(response("PRIMARY"))
    scoring = FakeClient(response("A", {"A": -0.1, "B": -2.3}))
    client = LlamaCppClient(
        model="qwen38-code",
        _inner=primary,
        _scoring_inner=scoring,
    )

    result = client.complete(
        [Message("user", "A, B, C or D?")],
        max_tokens=1,
        top_logprobs=4,
    )

    assert primary.chat.completions.calls == []
    assert len(scoring.chat.completions.calls) == 1
    call = scoring.chat.completions.calls[0]
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 4
    assert result.top_logprobs == {"A": -0.1, "B": -2.3}
