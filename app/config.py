"""Single source of configuration, service URLs and Spring-style profiles."""
from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

COINS = ("BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX")
FEATURES_C = ("Volatility20", "MaxPrice20", "MaxDrawdown", "GR")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)
    app_profile: Literal["local", "dev", "prod"] = "local"
    history_days: int = Field(default=90, ge=35, le=365)
    request_timeout_seconds: float = Field(default=8.0, gt=0, le=120)
    model_path: str = "models/predict_seven_day_trend.pkl"
    model_metadata_path: str = "models/model_metadata.json"
    github_url: str = ""

    # All external service base addresses live here; no endpoint URLs in adapters.py.
    kraken_url: str = "https://api.kraken.com/0/public/OHLC"
    coindesk_url: str = "https://data-api.coindesk.com/spot/v1/historical/days"
    cmc_url: str = "https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical"
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coindesk_api_key: str = ""
    cmc_api_key: str = ""
    coingecko_api_key: str = ""
    student_a_url: str = ""
    student_b_url: str = ""
    student_d_url: str = ""

    @property
    def market_backend(self) -> str:
        return "mock" if self.app_profile == "local" else "http"

    @property
    def feature_backend(self) -> str:
        return "http" if self.app_profile == "prod" else "mock"

    @property
    def model_backend(self) -> str:
        return "trained"

    @model_validator(mode="after")
    def check_profile(self):
        if self.app_profile == "prod" and not all((self.student_a_url, self.student_b_url, self.student_d_url)):
            raise ValueError("prod needs STUDENT_A_URL, STUDENT_B_URL and STUDENT_D_URL")
        # The CMC key is intentionally optional: a missing key triggers CoinGecko fallback.
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()
