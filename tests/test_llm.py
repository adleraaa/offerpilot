import json
import pytest
from pydantic import BaseModel
from offerpilot.store import db
from offerpilot.llm import (LLMClient, PermanentLLMError, SpendCapExceeded)

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
