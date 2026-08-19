import hashlib
import json
import os
import subprocess

import yaml


def load_config(path: str = "config.yaml", *, strict: bool = False) -> dict:
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(
                f"{path} not found. Copy config.example.yaml to {path} and "
                f"fill in your own company list before running this command.")
        print(f"[warn] {path} not found; falling back to config.example.yaml. "
              f"Its company slugs are placeholders and will not resolve.")
        path = "config.example.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
