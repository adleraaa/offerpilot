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
        if self._today_spend() >= self.cfg["daily_spend_cap_usd"]:
            raise SpendCapExceeded(
                f"daily cap {self.cfg['daily_spend_cap_usd']} reached")
        last_err = None
        for _attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg["model"],
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0)
            except Exception as e:  # SDK/network errors
                status = getattr(e, "status_code", None)
                if status in (429,) or (status is not None and status >= 500):
                    raise RetryableLLMError(str(e)) from e
                if "timeout" in str(e).lower():
                    raise RetryableLLMError(str(e)) from e
                raise PermanentLLMError(str(e)) from e
            self._record(node, run_id, resp.usage)
            content = resp.choices[0].message.content
            try:
                return schema.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
        raise PermanentLLMError(
            f"validation failed after 3 attempts: {last_err}")
