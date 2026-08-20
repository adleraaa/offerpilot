import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from offerpilot.store import db
from offerpilot.models import MatchResult
from offerpilot.llm import (AuthLLMError, LLMClient, PermanentLLMError,
                            SpendCapExceeded, RetryableLLMError, is_peak_hour,
                            price_usage)

# Off-peak DeepSeek rates from https://api-docs.deepseek.com/quick_start/pricing/
# (checked 2026-08-20). Peak is 2x; a cached prompt token is ~31x cheaper.
V4_PRICES = {
    "deepseek-v4-flash": {
        "input_cache_hit_per_mtok_usd": 0.007,
        "input_cache_miss_per_mtok_usd": 0.22,
        "output_per_mtok_usd": 0.66,
        "peak_multiplier": 2.0,
    }
}

CFG = {"model": "deepseek-v4-flash", "daily_spend_cap_usd": 2.0,
       "base_url": "https://api.deepseek.com",
       "prices": V4_PRICES}


class Out(BaseModel):
    answer: int


class FakeCompletion:
    """No `model` attribute on purpose: the server-said-nothing fallback."""

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
def off_peak(monkeypatch):
    """Pin the clock off-peak.

    `_estimate_cost` takes no `when`, so anything asserting an exact estimate
    would otherwise cost twice as much between 01:00-04:00 and 06:00-10:00
    UTC and the test would pass or fail depending on the hour CI ran. The
    windows themselves are pinned by test_peak_hours_match_the_published_windows.
    """
    import offerpilot.llm as llm_module
    monkeypatch.setattr(llm_module, "is_peak_hour", lambda when=None: False)


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
                 "VALUES('deepseek-v4-flash', 0, 0, 5.0)")
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


def test_cap_rechecked_between_attempts(conn, off_peak):
    # The cap is now checked with a pre-call estimate, so the cap has to be
    # large enough for attempt 1's *predicted* cost and small enough that
    # attempt 1's *actual* cost pushes the next prediction over the line.
    # "s"/"u" is 2 chars, so int(2/4) == 0 prompt tokens: the estimate is the
    # assumed 1k reply and nothing else.
    est = 1000 * 0.66 / 1e6
    # FakeCompletion reports no cache split, so all 100 prompt tokens are misses.
    actual = (100 * 0.22 + 50 * 0.66) / 1e6
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
    # FakeCompletion reports no model, so the configured name is recorded.
    assert row["model"] == "deepseek-v4-flash" and row["node"] == "match"
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


# --- repair turns and auth errors (Task 3) --------------------------------

class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {
            "content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10,
                                    "completion_tokens": 5})()
        self.model = "deepseek-v4-flash"


class _SeqClient:
    """Returns queued payloads; records every messages list it was given."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        self.calls.append(kw["messages"])
        return _Resp(self.payloads.pop(0))


def _ok_match(source_id="proj"):
    return json.dumps({
        "eligibility": "pass", "eligibility_reasons": [],
        "eligibility_evidence_excerpt": None,
        "skills_score": 20, "project_score": 10, "domain_score": 10,
        "seniority_score": 10, "preference_score": 10,
        "evidence": [{"source_id": source_id, "section": "",
                      "supporting_text": "x"}],
        "gaps": [], "uncertainties": [], "confidence": 0.7})


def test_validate_failure_triggers_a_repair_turn_then_succeeds(conn, cfg):
    client = _SeqClient([_ok_match("invented"), _ok_match("proj")])
    llm = LLMClient(conn, cfg, "k", client=client)

    def validate(m):
        bad = [e.source_id for e in m.evidence if e.source_id != "proj"]
        if bad:
            raise ValueError(f"unknown source_id: {bad}")

    result = llm.structured(node="match", run_id=None, system="s", user="u",
                            schema=MatchResult, validate=validate)
    assert result.evidence[0].source_id == "proj"
    assert len(client.calls) == 2
    repair_turn = client.calls[1][-1]
    assert repair_turn["role"] == "user"
    assert "unknown source_id" in repair_turn["content"]


def test_repair_turn_shows_the_model_its_own_rejected_reply(conn, cfg):
    """The corrective turn is useless without the reply it corrects."""
    client = _SeqClient([_ok_match("invented"), _ok_match("proj")])
    llm = LLMClient(conn, cfg, "k", client=client)

    def validate(m):
        if any(e.source_id != "proj" for e in m.evidence):
            raise ValueError("unknown source_id")

    llm.structured(node="match", run_id=None, system="s", user="u",
                   schema=MatchResult, validate=validate)
    second = client.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert "invented" in second[2]["content"]
    # The first call must not have been polluted by the repair turns.
    assert len(client.calls[0]) == 2


def test_schema_failure_also_gets_a_corrective_turn(conn, cfg):
    """Bad JSON was already retried, but blind -- the model was never told."""
    client = _SeqClient(["not json at all", '{"answer": 7}'])
    llm = LLMClient(conn, cfg, "k", client=client)
    out = llm.structured(node="match", run_id=None, system="s", user="u",
                         schema=Out)
    assert out.answer == 7
    assert len(client.calls) == 2
    assert client.calls[1][-1]["role"] == "user"
    assert "schema" in client.calls[1][-1]["content"].lower()


def test_validate_failing_three_times_is_permanent(conn, cfg):
    client = _SeqClient([_ok_match("bad")] * 3)
    llm = LLMClient(conn, cfg, "k", client=client)

    def validate(m):
        raise ValueError("still wrong")

    with pytest.raises(PermanentLLMError):
        llm.structured(node="match", run_id=None, system="s", user="u",
                       schema=MatchResult, validate=validate)
    assert len(client.calls) == 3


def test_401_raises_auth_error_not_generic_permanent(conn, cfg):
    class Boom:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            e = Exception("invalid api key")
            e.status_code = 401
            raise e

    llm = LLMClient(conn, cfg, "k", client=Boom())
    with pytest.raises(AuthLLMError):
        llm.structured(node="match", run_id=None, system="s", user="u",
                       schema=MatchResult)


def test_403_is_an_auth_error_too(conn, cfg):
    class Boom:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            e = Exception("forbidden")
            e.status_code = 403
            raise e

    llm = LLMClient(conn, cfg, "k", client=Boom())
    with pytest.raises(AuthLLMError):
        llm.structured(node="match", run_id=None, system="s", user="u",
                       schema=MatchResult)


def test_auth_error_is_a_permanent_error_subclass(conn, cfg):
    """Callers that only know PermanentLLMError must still stop the job."""
    assert issubclass(AuthLLMError, PermanentLLMError)


def test_repair_turn_bounds_the_reason_it_quotes(conn, cfg):
    """The rejection reason can carry model-written text and a multi-kilobyte
    ValidationError dump. It is quoted back inside a *user* turn, so it is
    truncated: bounded prompt growth, and a bounded quantity of untrusted
    text in the trusted role."""
    client = _SeqClient([_ok_match("x"), _ok_match("proj")])
    llm = LLMClient(conn, cfg, "k", client=client)

    def validate(m):
        if any(e.source_id != "proj" for e in m.evidence):
            raise ValueError("rejected: " + "A" * 5000)

    llm.structured(node="match", run_id=None, system="s", user="u",
                   schema=MatchResult, validate=validate)
    reason_turn = client.calls[1][-1]["content"]
    assert "rejected:" in reason_turn
    assert len(reason_turn) < 1000


# --- spend-ledger pricing (Task D) ----------------------------------------
# Grounded in a live DeepSeek probe on 2026-08-20: the request asked for
# model="deepseek-chat" and the response came back as "deepseek-v4-flash", with
# a usage object carrying prompt_cache_hit_tokens / prompt_cache_miss_tokens.

class _Usage:
    def __init__(self, hit=0, miss=0, completion=0):
        self.prompt_cache_hit_tokens = hit
        self.prompt_cache_miss_tokens = miss
        self.prompt_tokens = hit + miss
        self.completion_tokens = completion


def test_peak_hours_match_the_published_windows():
    def at(hour):
        return datetime(2026, 8, 20, hour, 30, tzinfo=timezone.utc)
    for hour in (1, 2, 3, 6, 7, 8, 9):
        assert is_peak_hour(at(hour)) is True, hour
    for hour in (0, 4, 5, 10, 11, 17, 23):
        assert is_peak_hour(at(hour)) is False, hour


def test_cache_hits_are_priced_far_below_cache_misses():
    off = datetime(2026, 8, 20, 17, 0, tzinfo=timezone.utc)
    all_hit = price_usage(_Usage(hit=1_000_000), "deepseek-v4-flash",
                          V4_PRICES, when=off)
    all_miss = price_usage(_Usage(miss=1_000_000), "deepseek-v4-flash",
                           V4_PRICES, when=off)
    assert all_hit == pytest.approx(0.007)
    assert all_miss == pytest.approx(0.22)
    assert all_miss > all_hit * 20


def test_peak_pricing_is_double_off_peak():
    usage = _Usage(miss=1_000_000, completion=1_000_000)
    off = price_usage(usage, "deepseek-v4-flash", V4_PRICES,
                      when=datetime(2026, 8, 20, 17, 0, tzinfo=timezone.utc))
    peak = price_usage(usage, "deepseek-v4-flash", V4_PRICES,
                       when=datetime(2026, 8, 20, 2, 0, tzinfo=timezone.utc))
    assert off == pytest.approx(0.22 + 0.66)
    assert peak == pytest.approx(off * 2)


def test_usage_without_cache_fields_falls_back_to_all_miss():
    """Other OpenAI-compatible endpoints do not report a cache split."""
    class Bare:
        prompt_tokens = 1_000_000
        completion_tokens = 0

    off = datetime(2026, 8, 20, 17, 0, tzinfo=timezone.utc)
    assert price_usage(Bare(), "deepseek-v4-flash", V4_PRICES,
                       when=off) == pytest.approx(0.22)


def test_prompt_tokens_unaccounted_by_the_split_are_charged_as_misses():
    """A split that does not add up must never under-charge.

    The ledger is a fuse: tokens the server did not attribute to the cache are
    charged at the expensive rate, so a zeroed or partial split cannot silently
    make a batch look free and keep the cap from ever firing.
    """
    class Partial:
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = 0
        prompt_tokens = 1_000_000
        completion_tokens = 0

    off = datetime(2026, 8, 20, 17, 0, tzinfo=timezone.utc)
    assert price_usage(Partial(), "deepseek-v4-flash", V4_PRICES,
                       when=off) == pytest.approx(0.22)


def test_ledger_records_the_served_model_not_the_requested_alias(conn):
    """deepseek-chat is an alias; the ledger must record what was served."""
    served = _Resp(_ok_match())
    served.model = "deepseek-v4-flash"

    class AliasClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            assert kw["model"] == "deepseek-chat"
            return served

    cfg = {"base_url": "x", "model": "deepseek-chat",
           "daily_spend_cap_usd": 5.0, "prices": V4_PRICES}
    llm = LLMClient(conn, cfg, "k", client=AliasClient())
    llm.structured(node="match", run_id=None, system="s", user="u",
                   schema=MatchResult)
    row = conn.execute("SELECT model FROM llm_usage").fetchone()
    assert row["model"] == "deepseek-v4-flash"


def test_unknown_served_model_falls_back_to_the_configured_prices(conn):
    """An unpriced model must not crash the batch with a KeyError."""
    served = _Resp(_ok_match())
    served.model = "deepseek-v9-unreleased"

    class OddClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kw):
            return served

    cfg = {"base_url": "x", "model": "deepseek-v4-flash",
           "daily_spend_cap_usd": 5.0, "prices": V4_PRICES}
    llm = LLMClient(conn, cfg, "k", client=OddClient())
    llm.structured(node="match", run_id=None, system="s", user="u",
                   schema=MatchResult)
    row = conn.execute("SELECT model, estimated_cost_usd FROM "
                       "llm_usage").fetchone()
    assert row["model"] == "deepseek-v9-unreleased"
    assert row["estimated_cost_usd"] > 0


def test_an_unpriced_configured_model_estimates_at_the_dearest_table(conn,
                                                                    off_peak):
    """The pre-call fuse must survive a model name the price table lacks.

    `deepseek-chat` is exactly that case: a configured alias that no price
    table lists. The estimate has to produce a number anyway, and it errs
    high -- the dearest table on file -- because under-estimating is the
    failure that lets a run blow through the cap.
    """
    prices = dict(V4_PRICES)
    prices["deepseek-v4-pro"] = {"input_cache_hit_per_mtok_usd": 0.022,
                                 "input_cache_miss_per_mtok_usd": 0.66,
                                 "output_per_mtok_usd": 1.98,
                                 "peak_multiplier": 2.0}
    cfg = {"base_url": "x", "model": "deepseek-chat",
           "daily_spend_cap_usd": 5.0, "prices": prices}
    llm = LLMClient(conn, cfg, "k", client=FakeSDK(['{"answer": 7}']))
    # 1000 assumed reply tokens at the pro table's output rate, not flash's.
    assert llm._estimate_cost("s", "u") == pytest.approx(1000 * 1.98 / 1e6)
