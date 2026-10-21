import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)


class Message(BaseModel):
    content: str


class Choice(BaseModel):
    message: Message


class Completion(BaseModel):
    choices: list[Choice]


@dataclass
class LlmClient:
    base_url: str
    model: str
    cache_dir: Path
    timeout: float = 900
    retries: int = 4

    def complete[T: BaseModel](self, system: str, user: str, model: type[T]) -> T:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        payload = json.dumps(body, ensure_ascii=False).encode()
        cached = self.cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"
        if cached.exists():
            return model.model_validate_json(cached.read_text(encoding="utf-8"))
        content = self._post(payload).choices[0].message.content
        answer = model.model_validate_json(content)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(content, encoding="utf-8")
        return answer

    def _post(self, payload: bytes) -> Completion:
        headers = {"Authorization": "Bearer local", "Content-Type": "application/json"}
        for attempt in range(self.retries):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions", content=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                return Completion.model_validate_json(response.content)
            except (httpx.HTTPError, ValueError) as error:
                delay = 2 ** (attempt + 1)
                log.warning("llm call failed (%s), retry in %ss", error, delay)
                time.sleep(delay)
        raise RuntimeError("llm call failed after retries")
