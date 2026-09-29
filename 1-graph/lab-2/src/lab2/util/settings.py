from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAB2_", env_file=".env", extra="ignore")

    root: Path = Path()
    books: tuple[str, ...] = ("tanaka1981", "stat3")
    chunk_target_tokens: int = 900
    chunk_min_tokens: int = 250
    chunk_hard_max_tokens: int = 1100
    graphrag_chunk_tokens: int = 1200
    tiktoken_encoding: str = "o200k_base"
    bge_tokenizer_path: Path = Path(
        "~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
    ).expanduser()
    embedding_url: str = "http://127.0.0.1:8011/v1/embeddings"
    embedding_model: str = "bge-m3"
    openai_base_url: str = "https://teadove.space:7999/proxy/openai/v1"
    openai_api_key: str = ""
    judge_model: str = "gpt-4.1"
    tfidf_window: int = 300
    dirty_graphrag_dir: Path = Path("../lab1-combined/output")
    clean_graphrag_dir: Path = Path("graphrag-clean/output")

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "results" / "metrics"

    @property
    def glossary_path(self) -> Path:
        return self.root / "configs" / "glossary.yaml"

    @property
    def cleaning_path(self) -> Path:
        return self.root / "configs" / "cleaning.yaml"

    @property
    def evaluation_path(self) -> Path:
        return self.root / "configs" / "evaluation.yaml"
