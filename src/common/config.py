"""Configuration management."""
import os
from typing import Any, Dict, Optional
from omegaconf import OmegaConf


class Config:
    """Unified configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        self._cfg = OmegaConf.create()
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, path: str):
        cfg = OmegaConf.load(path)
        self._cfg = OmegaConf.merge(self._cfg, cfg)

    def merge(self, other: "Config") -> "Config":
        new = Config()
        new._cfg = OmegaConf.merge(self._cfg, other._cfg)
        return new

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return OmegaConf.select(self._cfg, key)
        except Exception:
            return default

    def set(self, key: str, value: Any):
        OmegaConf.update(self._cfg, key, value)

    def to_dict(self) -> Dict:
        return OmegaConf.to_container(self._cfg, resolve=True)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            OmegaConf.save(self._cfg, f)

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        cfg = cls()
        cfg._cfg = OmegaConf.create(d)
        return cfg

    def __repr__(self):
        return OmegaConf.to_yaml(self._cfg)
