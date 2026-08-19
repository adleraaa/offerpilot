import json
import pytest
from pydantic import BaseModel
from offerpilot.store import db
from offerpilot.models import MatchResult
from offerpilot.llm import (LLMClient, PermanentLLMError, SpendCapExceeded,
                            RetryableLLMError)

CFG = {"model": "deepseek-chat", "daily_spend_cap_usd": 2.0,
       "base_url": "https://api.deepseek.com",
       "prices": {"deepseek-chat": {"input_per_mtok_usd": 0.27,
                                     "output_per_mtok_usd": 1.10}}}


class Out(BaseModel):
    answer: int


class FakeCompletion:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {
            "content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 100,
                                    "completion_tokens": 50})()


class FakeChat:
    def __init__(self, contents):
        self.contents = list(contents)
        self.completions = self

    def create(self, **kwargs):
        return FakeCompletion(self.contents.pop(0))


class FakeSDK:
    def __init__(self, contents):
        self.chat = FakeChat(contents)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    return c


def test_parses_valid_json(conn):
    cli = LLMClient(conn, CFG, "k", client=FakeSDK(['{"answer": 7}']))
    out = cli.structured(node="match", run_id=None, system="s", user="u",
                         schema=Out)
    assert out.answer == 7
    n = conn.execute("SELECT COUNT(*) c FROM llm_usage").fetchone()["c"]
    assert n == 1


def test_retries_then_permanent_on_bad_json(conn):
    cli = LLMClient(conn, CFG, "k",
                    client=FakeSDK(["nope", "still nope", '{"wrong": 1}']))
    with pytest.raises(PermanentLLMError):
        cli.structured(node="match", run_id=None, system="s", user="u",
                       schema=Out)
    n = conn.execute("SELECT COUNT(*) c FROM llm_usage").fetchone()["c"]
    assert n == 3


def test_spend_cap_blocks_new_calls(conn):
    conn.execute("INSERT INTO llm_usage(model, prompt_tokens, "
                 "completion_tokens, estimated_cost_usd) "
                 "VALUES('deepseek-chat', 0, 0, 5.0)")
    conn.commit()
    cli = LLMClient(conn, CFG, "k", client=FakeSDK(['{"answer": 1}']))
    with pytest.raises(SpendCapExceeded):
        cli.structured(node="match", run_id=None, system="s", user="u",
                       schema=Out)


class RaisingChat:
    def __init__(self, exc):
        self.exc = exc
        self.completions = self

    def create(self, **kwargs):
        raise self.exc


class RaisingSDK:
    def __init__(self, exc):
        self.chat = RaisingChat(exc)


class FakeAPITimeoutError(Exception):
    pass


class FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def test_429_maps_retryable(conn):
    cli = LLMClient(conn, CFG, "k", client=RaisingSDK(FakeStatusError(429)))
    with pytest.raises(RetryableLLMError):
        cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)


def test_500_maps_retryable(conn):
    cli = LLMClient(conn, CFG, "k", client=RaisingSDK(FakeStatusError(503)))
    with pytest.raises(RetryableLLMError):
        cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)


def test_timeout_type_maps_retryable(conn):
    cli = LLMClient(conn, CFG, "k", client=RaisingSDK(FakeAPITimeoutError("slow")))
    with pytest.raises(RetryableLLMError):
        cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)


def test_other_sdk_error_is_permanent(conn):
    cli = LLMClient(conn, CFG, "k", client=RaisingSDK(ValueError("bad request")))
    with pytest.raises(PermanentLLMError):
        cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)


def test_cap_rechecked_between_attempts(conn):
    # The cap is now checked with a pre-call estimate, so the cap has to be
    # large enough for attempt 1's *predicted* cost and small enough that
    # attempt 1's *actual* cost pushes the next prediction over the line.
    est = (2 / 4 * 0.27 + 1000 * 1.10) / 1e6      # estimate for system/user "s"/"u"
    actual = (100 * 0.27 + 50 * 1.10) / 1e6       # what FakeCompletion reports
    tight = dict(CFG, daily_spend_cap_usd=est + actual / 2)
    cli = LLMClient(conn, tight, "k",
                    client=FakeSDK(["nope", "nope", "nope"]))
    with pytest.raises(SpendCapExceeded):
        cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)
    n = conn.execute("SELECT COUNT(*) c FROM llm_usage").fetchone()["c"]
    assert n == 1  # attempt 1 ran and was recorded; attempt 2 blocked by cap


def test_usage_row_content(conn):
    cli = LLMClient(conn, CFG, "k", client=FakeSDK(['{"answer": 7}']))
    cli.structured(node="match", run_id=None, system="s", user="u", schema=Out)
    row = conn.execute("SELECT * FROM llm_usage").fetchone()
    assert row["model"] == "deepseek-chat" and row["node"] == "match"
    assert row["prompt_tokens"] == 100 and row["completion_tokens"] == 50
    assert row["estimated_cost_usd"] > 0


@pytest.fixture()
def cfg():
    return dict(CFG)


def test_malformed_usage_is_a_permanent_error_not_an_attribute_error(conn, cfg):
    """Response parsing must live inside the client's error mapping."""
    class NoUsage:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            return type("R", (), {"choices": [], "usage": None})()

    llm = LLMClient(conn, cfg, "k", client=NoUsage())
    with pytest.raises(PermanentLLMError):
        llm.structured(node="match", run_id=None, system="s", user="u",
                       schema=MatchResult)


def test_pre_call_estimate_blocks_a_call_that_would_breach_the_cap(conn, cfg):
    cfg = dict(cfg, daily_spend_cap_usd=0.0001)
    calls = []

    class Counting:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            calls.append(1)
            raise AssertionError("should never be called")

    llm = LLMClient(conn, cfg, "k", client=Counting())
    with pytest.raises(SpendCapExceeded):
        llm.structured(node="match", run_id=None, system="s" * 40000,
                       user="u" * 40000, schema=MatchResult)
    assert calls == []


def test_usage_none_with_a_valid_choice_is_a_permanent_error(conn, cfg):
    """Reaches _record: choices are fine, usage is not.

    This is the case that pins _record's placement *inside* the try. With
    non-empty choices the IndexError never fires, so the only thing that can
    raise is usage.prompt_tokens inside _record -- and it must be mapped to
    PermanentLLMError, not leak an AttributeError into the batch loop.
    """
    class R:
        choices = [type("C", (), {"message": type("M", (), {
            "content": '{"answer": 1}'})()})()]
        usage = None

    class Sdk:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            return R()

    llm = LLMClient(conn, cfg, "k", client=Sdk())
    with pytest.raises(PermanentLLMError):
        llm.structured(node="match", run_id=None, system="s", user="u",
                       schema=Out)


def test_estimate_scales_with_prompt_length(conn, cfg):
    """The input-token half of the estimate must actually count the prompt.

    The cap test above trips on the fixed 1k-reply assumption alone, so this
    is the only thing pinning the prompt-size term.
    """
    llm = LLMClient(conn, cfg, "k", client=FakeSDK(['{"answer": 7}']))
    small = llm._estimate_cost("s", "u")
    big = llm._estimate_cost("s" * 40000, "u" * 40000)
    assert big > small * 5
