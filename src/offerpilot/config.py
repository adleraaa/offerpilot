import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
