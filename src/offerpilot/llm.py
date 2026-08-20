import json
from pydantic import BaseModel, ValidationError


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

    def _estimate_cost(self, system: str, user: str) -> float:
        prices = self.cfg["prices"][self.cfg["model"]]
        # ~4 chars per token is the standard rough ratio; assume a 1k reply.
        prompt_tokens = (len(system) + len(user)) / 4
        return (prompt_tokens * prices["input_per_mtok_usd"]
                + 1000 * prices["output_per_mtok_usd"]) / 1e6

    def _record(self, node, run_id, usage):
        prices = self.cfg["prices"][self.cfg["model"]]
        cost = (usage.prompt_tokens * prices["input_per_mtok_usd"]
                + usage.completion_tokens * prices["output_per_mtok_usd"]) / 1e6
        self.conn.execute(
            "INSERT INTO llm_usage(run_id, node, model, prompt_tokens, "
            "completion_tokens, estimated_cost_usd) VALUES(?,?,?,?,?,?)",
            (run_id, node, self.cfg["model"], usage.prompt_tokens,
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
                self._record(node, run_id, usage)
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
