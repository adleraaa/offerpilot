import json
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

# DeepSeek charges double during two published windows, and prices cached
# prompt tokens ~31x below uncached ones. Both matter here: OfferPilot resends
# an identical system prompt plus the whole profile for every job in a batch,
# so some prompt tokens come back as hits. Do not assume most of them do -- in
# the 2026-08-20 smoke run only 21.9% of prompt tokens were billed at the hit
# rate, in 1024-token blocks, and three of the six calls were billed entirely
# at the miss rate. That is why the fallback below charges every unattributed
# token at the miss rate rather than guessing. Rates and windows:
# https://api-docs.deepseek.com/quick_start/pricing/ (checked
# 2026-08-20); the numbers themselves live in config, only the shape is here.
_PEAK_HOURS_UTC = frozenset({1, 2, 3, 6, 7, 8, 9})


def is_peak_hour(when: datetime | None = None) -> bool:
    """DeepSeek peak windows: 01:00-04:00 and 06:00-10:00 UTC."""
    now = when or datetime.now(timezone.utc)
    return now.hour in _PEAK_HOURS_UTC


def price_usage(usage, model: str, prices: dict, *, when=None) -> float:
    """Cost one call. Splits cached vs uncached prompt tokens when reported.

    The single costing function: the pre-call estimate and the post-call
    ledger row both go through it, because two formulas drift apart and the
    fuse then stops matching the thing it is fusing.
    """
    table = prices[model]
    multiplier = table.get("peak_multiplier", 2.0) if is_peak_hour(when) else 1.0

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None or miss is None or hit + miss < prompt:
        # Endpoints that report no cache split, or one that does not add up:
        # charge every unattributed prompt token at the dearer miss rate. The
        # ledger is a fuse, so under-charging is the dangerous direction.
        hit = hit or 0
        miss = max(prompt - hit, 0)

    completion = getattr(usage, "completion_tokens", 0) or 0
    total = (hit * table["input_cache_hit_per_mtok_usd"]
             + miss * table["input_cache_miss_per_mtok_usd"]
             + completion * table["output_per_mtok_usd"])
    return total * multiplier / 1e6


# A rejection reason is quoted back to the model inside a *user* turn, and it
# is not fully trusted text: a pydantic ValidationError echoes the offending
# values, and a validator can name model-written ids, both ultimately derived
# from an untrusted posting. So it is collapsed to one line and clipped -- an
# attacker gets a few hundred characters in the trusted role at most, and the
# prompt cannot grow by kilobytes per repair turn.
_REASON_MAX = 300


def _reason(err: Exception) -> str:
    text = " ".join(str(err).split())
    return text if len(text) <= _REASON_MAX else text[:_REASON_MAX] + "..."


class RetryableLLMError(Exception):
    pass


class PermanentLLMError(Exception):
    pass


class AuthLLMError(PermanentLLMError):
    """Bad or missing credentials -- abort the batch, do not burn jobs.

    Every job in the queue would fail identically on a rejected key, so this
    is the one permanent error a caller must not treat per-job. It subclasses
    PermanentLLMError so a caller that has not been taught about it still
    stops the job rather than retrying forever; callers that have must catch
    it *before* PermanentLLMError.
    """


class SpendCapExceeded(Exception):
    pass


class LLMClient:
    def __init__(self, conn, llm_config: dict, api_key: str, client=None):
        self.conn = conn
        self.cfg = llm_config
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=llm_config["base_url"])
        self.client = client

    def _today_spend(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd),0) s FROM llm_usage "
            "WHERE created_at >= date('now')").fetchone()
        return row["s"]

    def _price_key(self, model: str) -> str:
        """Which price table to charge `model` against.

        Two names can miss the table. The configured one may be an alias --
        a 2026-08-20 probe asked for `deepseek-chat` and was served
        `deepseek-v4-flash` -- and the served one may be newer than the
        config. Neither may raise: an unpriced model must still produce a
        number, so the fallback is the configured table and, failing that,
        the dearest table on file. Erring high is the safe direction for a
        spending fuse.
        """
        prices = self.cfg["prices"]
        if model in prices:
            return model
        if self.cfg["model"] in prices:
            return self.cfg["model"]
        if not prices:
            raise KeyError("llm.prices is empty; no rate to charge against")
        return max(prices, key=lambda m: prices[m]["output_per_mtok_usd"])

    def _estimate_cost(self, system: str, user: str) -> float:
        """Pessimistic pre-call estimate: every prompt token a cache miss.

        The estimate is a fuse, not an invoice -- it runs before the call, so
        it cannot know the cache split the server will report. Assuming the
        expensive case means the cap fires early rather than late.
        """
        class _Est:
            # ~4 chars per token is the standard rough ratio; assume a 1k reply.
            prompt_cache_hit_tokens = 0
            prompt_cache_miss_tokens = int((len(system) + len(user)) / 4)
            prompt_tokens = prompt_cache_miss_tokens
            completion_tokens = 1000

        return price_usage(_Est(), self._price_key(self.cfg["model"]),
                           self.cfg["prices"])

    def _record(self, node, run_id, usage, served_model: str | None = None):
        """Write one ledger row, keyed on the model the server actually served.

        Recording the requested name would price the wrong model and put a
        name in `llm_usage.model` that never ran, which is also what the
        `runs` / `run_steps` reproducibility trail depends on being true.
        """
        model = served_model or self.cfg["model"]
        cost = price_usage(usage, self._price_key(model), self.cfg["prices"])
        self.conn.execute(
            "INSERT INTO llm_usage(run_id, node, model, prompt_tokens, "
            "completion_tokens, estimated_cost_usd) VALUES(?,?,?,?,?,?)",
            (run_id, node, model, usage.prompt_tokens,
             usage.completion_tokens, cost))
        self.conn.commit()

    def structured(self, *, node: str, run_id, system: str, user: str,
                   schema: type[BaseModel], validate=None) -> BaseModel:
        """Ask for one JSON object and return it parsed.

        `validate` rejects a reply that parses but is semantically wrong, by
        raising ValueError. Both kinds of rejection -- unparseable and
        rejected -- buy a *repair turn*: the offending reply and the reason
        are appended to the conversation and the model is asked again, inside
        the same 3-attempt budget. Retrying without telling the model what was
        wrong just re-rolls the same dice.
        """
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_err = None
        for _attempt in range(3):
            cap = self.cfg["daily_spend_cap_usd"]
            spent = self._today_spend()
            if spent + self._estimate_cost(system, user) > cap:
                raise SpendCapExceeded(
                    f"daily cap {cap} would be exceeded (spent {spent:.4f})")
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg["model"],
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0)
                usage = resp.usage
                content = resp.choices[0].message.content
                self._record(node, run_id, usage,
                             getattr(resp, "model", None))
            except Exception as e:  # SDK/network + malformed-response errors
                status = getattr(e, "status_code", None)
                name = type(e).__name__
                # Checked first: a rejected key fails every job identically,
                # so the caller aborts the batch instead of burning it.
                if status in (401, 403):
                    raise AuthLLMError(str(e)) from e
                if status == 429 or (isinstance(status, int) and status >= 500):
                    raise RetryableLLMError(str(e)) from e
                if isinstance(e, TimeoutError) or "Timeout" in name or "Connection" in name:
                    raise RetryableLLMError(str(e)) from e
                raise PermanentLLMError(str(e)) from e
            try:
                parsed = schema.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                messages = self._repair(
                    messages, content,
                    f"Your previous reply did not match the required schema: "
                    f"{_reason(e)}. Reply again with ONLY a valid JSON object "
                    f"that satisfies it.")
                continue
            if validate is not None:
                try:
                    validate(parsed)
                except ValueError as e:
                    last_err = e
                    messages = self._repair(
                        messages, content,
                        f"Your previous reply was rejected: {_reason(e)} Fix "
                        f"only that problem and reply again with ONLY a valid "
                        f"JSON object.")
                    continue
            return parsed
        raise PermanentLLMError(
            f"validation failed after 3 attempts: {last_err}")

    @staticmethod
    def _repair(messages: list[dict], content, instruction: str) -> list[dict]:
        """Append the rejected reply and the correction as a new turn.

        Returns a new list: the caller's previous `messages` may already have
        been handed to the SDK, and mutating it in place would rewrite what
        that call was given.
        """
        return messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": instruction}]
