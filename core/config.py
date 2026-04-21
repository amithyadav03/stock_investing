import os
import functools
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Primary AI — Claude (Anthropic)
    ANTHROPIC_API_KEY: str = ""

    # Claude model selection
    CLAUDE_SONNET_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"

    # Legacy / optional fallback
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Kite Connect (Zerodha)
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Telegram webhook secret — set this to any random string in .env
    # and configure same value in Telegram setWebhook secret_token
    TELEGRAM_WEBHOOK_SECRET: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///./db/trading.db"
    CHROMA_DB_DIR: str = "./db/chroma_memory"

    # Ngrok (local Telegram webhook tunneling)
    NGROK_AUTH_TOKEN: str = ""

    # Langfuse observability (optional)
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = ""

    @functools.cached_property
    def strategy(self):
        """Loads strategy_config.yaml once and caches it for the process lifetime."""
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "strategy_config.yaml")
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            print("[Config] strategy_config.yaml loaded.")
            return data
        except Exception as e:
            print(f"[Config] Error loading strategy_config.yaml: {e}")
            return {}

    @property
    def PAPER_MODE(self) -> bool:
        return self.strategy.get("capital", {}).get("paper_mode", True)

    @property
    def PAPER_CAPITAL(self) -> float:
        return float(self.strategy.get("capital", {}).get("paper_capital", 500000))

    def validate_critical_keys(self):
        """Warn at startup about missing critical configuration."""
        issues = []
        if not self.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY not set — AI analysis will be skipped.")
        if not self.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN not set — Telegram notifications disabled.")
        if not self.KITE_API_KEY or not self.KITE_ACCESS_TOKEN:
            issues.append("KITE_API_KEY/KITE_ACCESS_TOKEN not set — live orders disabled.")
        if not self.TELEGRAM_WEBHOOK_SECRET:
            issues.append("TELEGRAM_WEBHOOK_SECRET not set — webhook is unauthenticated (dev only).")
        mode = "PAPER" if self.PAPER_MODE else "LIVE"
        cap = self.PAPER_CAPITAL if self.PAPER_MODE else self.strategy.get("capital", {}).get("live_capital", 0)
        print(f"[Config] Mode: {mode} | Capital: ₹{cap:,.0f}")
        for issue in issues:
            print(f"[Config] WARNING: {issue}")
        return len(issues) == 0

    class Config:
        env_file = ".env"


settings = Settings()
