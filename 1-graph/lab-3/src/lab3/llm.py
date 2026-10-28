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


def parse_json[T: BaseModel](model: type[T], content: str) -> T:
    text = content.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return model.model_validate_json(text)
    candidate, _ = json.JSONDecoder().raw_decode(text[start:])
    return model.model_validate(candidate)


@dataclass
class LlmClient:
    base_url: str
    model: str
    cache_dir: Path
    api_key: str = "local"
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    timeout: float = 900
    retries: int = 4

    def complete[T: BaseModel](self, system: str, user: str, model: type[T]) -> T:
        body = self._body(system, user, json_mode=True)
        payload = json.dumps(body, ensure_ascii=False).encode()
        cached = self.cache_dir / f"{self._key(payload)}.json"
        if cached.exists():
            return parse_json(model, cached.read_text(encoding="utf-8"))
        try:
            content = self._call(payload)
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in (400, 404, 422):
                raise
            log.warning("json mode rejected (%s), retrying without response_format", error)
            body.pop("response_format", None)
            content = self._call(json.dumps(body, ensure_ascii=False).encode())
        answer = parse_json(model, content)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(content, encoding="utf-8")
        return answer

    def _body(self, system: str, user: str, json_mode: bool) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _key(self, payload: bytes) -> str:
        return hashlib.sha256(self.base_url.encode() + b"\0" + payload).hexdigest()

    def _headers(self) -> dict[str, str]:
        value = f"{self.api_key_prefix}{self.api_key}" if self.api_key_prefix else self.api_key
        return {"Content-Type": "application/json", self.api_key_header: value}

    def _call(self, payload: bytes) -> str:
        return self._post(payload).choices[0].message.content

    def _post(self, payload: bytes) -> Completion:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions", content=payload, headers=self._headers(), timeout=self.timeout
                )
                response.raise_for_status()
                return Completion.model_validate_json(response.content)
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500 and error.response.status_code not in (408, 429):
                    raise
                last = error
            except (httpx.HTTPError, ValueError) as error:
                last = error
            if attempt < self.retries - 1:
                delay = 2 ** (attempt + 1)
                log.warning("llm call failed (%s), retry in %ss", last, delay)
                time.sleep(delay)
        raise RuntimeError("llm call failed after retries") from last
