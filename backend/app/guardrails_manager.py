from pathlib import Path
from nemoguardrails import LLMRails,RailsConfig
GUARDRAILS_PATH=Path(__file__).parent/"guardrails"
config=RailsConfig.from_path(str(GUARDRAILS_PATH))
rails=LLMRails(config)