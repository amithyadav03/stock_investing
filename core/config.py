import os
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    # Kite
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    
    # Database
    DATABASE_URL: str = "sqlite:///./db/trading.db"
    CHROMA_DB_DIR: str = "./db/chroma_memory"

    # Ngrok (for local Telegram webhook tunneling)
    NGROK_AUTH_TOKEN: str = ""
    
    # Langfuse
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = ""

    @property
    def strategy(self):
        """Loads and parses the dynamic strategy_config.yaml file."""
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "strategy_config.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Error loading strategy_config.yaml: {e}")
            return {}

    class Config:
        env_file = ".env"

settings = Settings()
