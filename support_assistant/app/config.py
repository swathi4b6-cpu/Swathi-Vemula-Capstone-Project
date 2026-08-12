import os

class Settings:
    # 1 = Fully offline rule-based mock mode (Graded Baseline)
    # 0 = Call downstream production LLM cloud endpoints
    MOCK_LLM: int = int(os.getenv("MOCK_LLM", "1"))
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()
