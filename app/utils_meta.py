# app/utils_meta.py
import json, os

def load_meta(workdir: str):
    path = os.path.join(workdir, "meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_meta(workdir: str, meta: dict):
    path = os.path.join(workdir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
