"""Configuration resolution for Ollama Chat Streamer.

Resolves settings from CLI arguments, environment variables, and YAML config files
using a consistent precedence: CLI > env var > YAML > default.
"""

import os
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration from a file path."""
    if not config_path:
        return {}
    if not HAS_YAML:
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def _yaml_get(data: Dict[str, Any], *keys: str) -> Any:
    """Traverse nested dict by keys, returning None if path doesn't exist."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_float(
    cli_value: Optional[float],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: float
) -> float:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            return default
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        try:
            return float(yaml_value)
        except (ValueError, TypeError):
            return default
    return default


def _resolve_bool(
    cli_true: bool,
    cli_false: bool,
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: bool
) -> bool:
    if cli_true:
        return True
    if cli_false:
        return False
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        return bool(yaml_value)
    return default


def _resolve_str(
    cli_value: Optional[str],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: str
) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        return str(yaml_value)
    return default


def _resolve_int(
    cli_value: Optional[int],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: int
) -> int:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            return default
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        try:
            return int(yaml_value)
        except (ValueError, TypeError):
            return default
    return default


def _resolve_think(
    cli_value: Optional[str],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: str = "auto"
) -> str:
    """Resolve the 'think' setting: auto, true, or false."""
    if cli_value is not None:
        return cli_value.lower()
    env_value = os.environ.get(env_name, "").lower()
    if env_value in ("true", "1", "yes", "on"):
        return "true"
    if env_value in ("false", "0", "no", "off"):
        return "false"
    if env_value == "auto":
        return "auto"
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        v = str(yaml_value).lower()
        if v in ("true", "1", "yes", "on"):
            return "true"
        if v in ("false", "0", "no", "off"):
            return "false"
        if v == "auto":
            return "auto"
    return default