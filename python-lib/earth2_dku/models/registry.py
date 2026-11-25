"""Model registry backed by models.yaml."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from ..errors import ConfigError

REGISTRY_PATH = Path(__file__).with_name("models.yaml")
_SUPPORT_VALUES = ("true", "false", "unverified")
_VARIABLES_CACHE: dict[str, list[str]] = {}


def _normalize_support(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text not in _SUPPORT_VALUES:
        raise ConfigError(
            f"supports_variables_param must be one of {_SUPPORT_VALUES}, got '{value}'. Fix models.yaml."
        )
    return text


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    nim_image: str
    input_variables_from: str
    n_channels: int
    step_hours: int
    min_gpu_mem_gb: int
    supports_variables_param: str
    grid: str

    @property
    def nim_version(self) -> str:
        return self.nim_image.rsplit(":", 1)[-1] if ":" in self.nim_image else self.nim_image

    def input_variables(self) -> list[str]:
        """Model input variable list, imported lazily from Earth2Studio.

        Raises ConfigError when Earth2Studio is missing or its list length does not
        match ``n_channels`` (catches Earth2Studio version drift).
        """
        if self.input_variables_from not in _VARIABLES_CACHE:
            module_path, _, attr = self.input_variables_from.partition(":")
            try:
                module = importlib.import_module(module_path)
                variables = list(getattr(module, attr))
            except (ImportError, AttributeError) as exc:
                raise ConfigError(
                    f"Cannot load the input variable list for {self.model_id} from "
                    f"'{self.input_variables_from}' ({exc}). Check that the plugin code environment "
                    "is built and its Earth2Studio version provides this model."
                ) from exc
            _VARIABLES_CACHE[self.input_variables_from] = variables
        variables = list(_VARIABLES_CACHE[self.input_variables_from])
        if len(variables) != self.n_channels:
            raise ConfigError(
                f"{self.model_id} expects {self.n_channels} input channels but "
                f"{self.input_variables_from} lists {len(variables)}. The installed Earth2Studio version "
                "does not match this NIM; pin a compatible earth2studio version in the plugin code env."
            )
        return variables


def _spec_from_entry(model_id: str, entry: dict) -> ModelSpec:
    try:
        return ModelSpec(
            model_id=model_id,
            label=str(entry["label"]),
            nim_image=str(entry["nim_image"]),
            input_variables_from=str(entry["input_variables_from"]),
            n_channels=int(entry["n_channels"]),
            step_hours=int(entry["step_hours"]),
            min_gpu_mem_gb=int(entry["min_gpu_mem_gb"]),
            supports_variables_param=_normalize_support(entry["supports_variables_param"]),
            grid=str(entry["grid"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Model '{model_id}' in models.yaml is missing key {exc}.") from exc


def load_registry(path=None) -> dict[str, ModelSpec]:
    with open(path or REGISTRY_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {model_id: _spec_from_entry(model_id, entry) for model_id, entry in raw.items()}


@lru_cache(maxsize=1)
def _default_registry() -> dict[str, ModelSpec]:
    return load_registry()


def list_models() -> list[ModelSpec]:
    return list(_default_registry().values())


def get_model(model_id: str) -> ModelSpec:
    registry = _default_registry()
    if model_id not in registry:
        raise ConfigError(
            f"Unknown model '{model_id}'. Available models: {', '.join(registry)}. "
            "Pick one of these in the connection preset."
        )
    return registry[model_id]
