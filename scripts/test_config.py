
from pathlib import Path

from seven.core.config import (
    AppConfig,
    ConfigurationManager,
)

manager = ConfigurationManager(
    config_path=Path("configs/app.yaml"),
    schema=AppConfig,
    dotenv_path=Path(".env"),
)

config = manager.load()

print("SEVEN CONFIGURATION LOADED")
print("Application:", config.app.name)
print("Version:", config.app.version)
print("Environment:", config.app.environment)
print("AI Provider:", manager.get("ai.provider"))
print("AI Model:", manager.get("ai.model"))
print("Voice Enabled:", manager.get("voice.enabled"))
