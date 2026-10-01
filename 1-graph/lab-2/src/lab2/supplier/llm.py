import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel

from lab2.util.text import count_tokens

log = logging.getLogger(__name__)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Message(BaseModel):
    content: str


class Choice(BaseModel):
    message: Message


class Completion(BaseModel):
    choices: list[Choice]
    usage: Usage = Usage()


@dataclass
class LlmClient:
    base_url: str
    api_key: str
    model: str
    cache_dir: Path
    max_request_bytes: int = 18000
    retries: int = 6
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0
    request_tokens: int = 0

    def complete_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        payload = json.dumps(body, ensure_ascii=False).encode()
        if len(payload) > self.max_request_bytes:
            raise ValueError(f"request of {len(payload)} bytes exceeds the proxy limit")
        key = hashlib.sha256(payload).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        self.calls += 1
        self.request_tokens += count_tokens(system) + count_tokens(user)
        if cached.exists():
            self.cached_calls += 1
            return json.loads(cached.read_text(encoding="utf-8"))
        completion = self._post(payload)
        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens
        result = json.loads(completion.choices[0].message.content)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    def _post(self, payload: bytes) -> Completion:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        for attempt in range(self.retries):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions", content=payload, headers=headers, timeout=90)
                response.raise_for_status()
                return Completion.model_validate_json(response.content)
            except (httpx.HTTPError, ValueError) as error:
                delay = min(2 ** (attempt + 1), 60)
                log.warning("llm call failed (%s), retry in %ss", error, delay)
                time.sleep(delay)
        raise RuntimeError("llm call failed after retries")
