import time
from typing import List, Dict, Any, Generator
from openai import OpenAI
from src.config import settings

class NemotronLLMClient:
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def api_key(self) -> str:
        return self._api_key or settings.OPENROUTER_API_KEY

    @property
    def model(self) -> str:
        return self._model or settings.OPENROUTER_MODEL

    @property
    def base_url(self) -> str:
        return self._base_url or settings.OPENROUTER_BASE_URL

    @property
    def client(self) -> OpenAI:
        key = self.api_key or "sk-dummy-key"
        if self._client is None or getattr(self._client, "api_key", "") != key:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=key,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Enterprise RAG Assistant"
                }
            )
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Generate full completion with retry and latency/token tracking."""
        start_time = time.time()
        last_err = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response and hasattr(response, "choices") and response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    usage = getattr(response, "usage", None)
                    latency = time.time() - start_time
                    
                    return {
                        "content": content,
                        "latency_seconds": round(latency, 3),
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "model": self.model
                    }
                else:
                    time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
                
        return {
            "content": f"⚠️ [Generation Error] Unable to generate grounded response due to upstream LLM provider unavailability ({last_err}). Please verify API status or retry shortly.",
            "latency_seconds": round(time.time() - start_time, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": self.model,
            "error": str(last_err)
        }

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream completion tokens with robust exception handling."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"⚠️ [Streaming Error: {e}]"

llm_client = NemotronLLMClient()
