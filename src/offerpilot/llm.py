import json
from pydantic import BaseModel, ValidationError


class RetryableLLMError(Exception):
    pass


class PermanentLLMError(Exception):
    pass


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
                   schema: type[BaseModel]) -> BaseModel:
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
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0)
                usage = resp.usage
                content = resp.choices[0].message.content
                self._record(node, run_id, usage)
            except Exception as e:  # SDK/network + malformed-response errors
                status = getattr(e, "status_code", None)
                name = type(e).__name__
                if status == 429 or (isinstance(status, int) and status >= 500):
                    raise RetryableLLMError(str(e)) from e
                if isinstance(e, TimeoutError) or "Timeout" in name or "Connection" in name:
                    raise RetryableLLMError(str(e)) from e
                raise PermanentLLMError(str(e)) from e
            try:
                return schema.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
        raise PermanentLLMError(
            f"validation failed after 3 attempts: {last_err}")
