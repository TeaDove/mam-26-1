import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = None


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingItem]
    usage: Usage


def build_app(model_name: str) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    app = FastAPI()

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        texts = [request.input] if isinstance(request.input, str) else request.input
        vectors = encoder.encode(texts, normalize_embeddings=True, batch_size=32)
        tokens = sum(len(encoder.tokenizer.encode(text)) for text in texts)
        return EmbeddingResponse(
            model=model_name,
            data=[EmbeddingItem(index=i, embedding=vector.tolist()) for i, vector in enumerate(vectors)],
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name, "device": device}

    return app


def main() -> None:
    app = build_app(os.environ.get("EMBED_MODEL", "BAAI/bge-m3"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("EMBED_PORT", "8011")), log_level="warning")


if __name__ == "__main__":
    main()
