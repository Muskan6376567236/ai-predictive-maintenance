"""Configuration management for the predictive maintenance system."""

import os
from pathlib import Path
from typing import Optional

import yaml


class Config:
    """Configuration manager for loading and accessing settings."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or environment."""
        self._config = {}

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)

        # Override with environment variables
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "DB_HOST": ("database", "host"),
            "DB_PORT": ("database", "port"),
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
            "LOG_LEVEL": ("logging", "level"),
        }

        for env_key, config_path in env_mappings.items():
            value = os.getenv(env_key)
            if value:
                self.set_nested(config_path, value)

    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set_nested(self, keys: tuple, value):
        """Set a nested configuration value."""
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value