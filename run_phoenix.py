import sys
import io
import uvicorn
from openinference.instrumentation.openai import OpenAIInstrumentor

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("Starting Arize Phoenix Observability Server via Uvicorn on 6006...")
print("=" * 60)

try:
    OpenAIInstrumentor().instrument()
    print("[OK] OpenInference OpenAI instrumentor enabled.")
except Exception as e:
    print(f"Warning on instrumentation: {e}")

from phoenix.server.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006, log_level="info")
