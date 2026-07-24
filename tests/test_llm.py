import json
import pytest
from pydantic import BaseModel
from offerpilot.store import db
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
    tight = dict(CFG, daily_spend_cap_usd=0.00005)
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
