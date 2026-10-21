from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAB3_", env_file=".env", extra="ignore")

    root: Path = Path()
    lab2_dir: Path = Path("../lab-2")
    graphs: tuple[str, ...] = ("dirty", "clean")
    encoders: tuple[str, ...] = ("gat", "sage")
    embedding_model: str = "BAAI/bge-m3"
    description_chars: int = 1000
    device: str = "cuda"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:14b-instruct"
    judge_model: str = "qwen2.5:14b-instruct"
    llm_timeout: float = 900
    gnn_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hybrid_weight: float = 0.3
    source_units: int = 2
    source_chars: int = 5000
    hidden: int = 64
    heads: int = 8
    dropout: float = 0.3
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 30
    top_k: int = 5
    seed_vertices: int = 3
    context_vertices: int = 30
    relation_bonus: float = 0.1
    two_hop_bonus: float = 0.05

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "results" / "metrics"

    @property
    def questions_path(self) -> Path:
        return self.root / "configs" / "questions.yaml"

    def graphrag_output(self, graph: str) -> Path:
        return (self.root / self.lab2_dir / f"graphrag-{graph}" / "output").resolve()

    @property
    def retrievers(self) -> tuple[str, ...]:
        return ("bge-m3", *self.encoders, "hybrid")
