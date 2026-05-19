import os
from src.utils.config import Config

def test_config_load():
    """Test basic configuration loading."""
    config = Config()
    assert config is not None
    # Check a default value
    assert config.get("api.port", 8000) == 8000

def test_config_env_override():
    """Test that environment variables override config."""
    os.environ["DB_HOST"] = "test_host"
    config = Config()
    assert config.get("database.host") == "test_host"
