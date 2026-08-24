import os
from src.config import settings

_phoenix_session = None

def setup_observability():
    """Initializes Arize Phoenix local zero-docker tracing."""
    global _phoenix_session
    try:
        import phoenix as px
        from openinference.instrumentation.openai import OpenAIInstrumentor
        
        # Instrument OpenAI calls (OpenRouter uses OpenAI client)
        OpenAIInstrumentor().instrument()
        
        # Launch in-memory Phoenix local server
        if _phoenix_session is None:
            _phoenix_session = px.launch_app(port=settings.PHOENIX_PORT)
            print(f"\n [OBSERVABILITY] Arize Phoenix running at http://localhost:{settings.PHOENIX_PORT}")
            
        return f"http://localhost:{settings.PHOENIX_PORT}"
    except Exception as e:
        print(f"[OBSERVABILITY WARNING] Could not start Phoenix: {e}")
        return "http://localhost:6006"

def get_phoenix_url() -> str:
    return f"http://localhost:{settings.PHOENIX_PORT}"
