from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app import config


@pytest.mark.parametrize("profile,market,features", [
    ("local", "mock", "mock"), ("dev", "http", "mock"), ("prod", "http", "http"),
])
def test_profiles(settings, profile, market, features):
    cfg = settings(app_profile=profile, student_a_url="https://a.test",
                   student_b_url="https://b.test", student_d_url="https://d.test")
    assert (cfg.market_backend, cfg.feature_backend, cfg.model_backend) == (market, features, "trained")


@pytest.mark.parametrize("kwargs", [
    {"app_profile": "prod"}, {"history_days": 34}, {"history_days": 366},
    {"request_timeout_seconds": 0}, {"request_timeout_seconds": 121}, {"app_profile": "unknown"},
])
def test_invalid_settings(settings, kwargs):
    with pytest.raises(ValidationError):
        settings(**kwargs)


def test_settings_are_cached(monkeypatch, settings):
    cfg = settings()
    constructor = Mock(return_value=cfg)
    monkeypatch.setattr(config, "Settings", constructor)
    assert config.get_settings() is cfg
    assert config.get_settings() is cfg
    constructor.assert_called_once_with()


def test_environment_and_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("APP_PROFILE=dev\nHISTORY_DAYS=50\nGITHUB_URL=\n")
    monkeypatch.setenv("HISTORY_DAYS", "60")
    cfg = config.Settings(_env_file=env)
    assert cfg.app_profile == "dev" and cfg.history_days == 60
    assert cfg.github_url.startswith("https://github.com/")
