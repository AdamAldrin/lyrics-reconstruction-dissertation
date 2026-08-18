from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_experiment.json"


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def load_config(config_path: str | None = None) -> dict:
    path = resolve_path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path)
    return config


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def slugify(text: str) -> str:
    value = clean_text(text).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "run"


def get_run_id(config: dict, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return slugify(explicit_run_id)

    configured = clean_text(config.get("run_id", ""))
    if configured:
        return slugify(configured)

    experiment_name = clean_text(config.get("experiment_name", "experiment"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(experiment_name)}-{timestamp}"


def get_run_dir(run_id: str) -> Path:
    return ensure_directory(PROJECT_ROOT / "outputs" / "runs" / run_id)


def load_template(path_like: str | Path) -> str:
    return resolve_path(path_like).read_text(encoding="utf-8")


def render_template(template_text: str, mapping: dict) -> str:
    safe_mapping = SafeDict({key: clean_text(value) for key, value in mapping.items()})
    return template_text.format_map(safe_mapping).strip()


def get_prompt_variants(config: dict) -> list[dict]:
    variants = config.get("prompt_variants", [])
    if not variants:
        raise ValueError("Config is missing prompt_variants.")
    return variants
