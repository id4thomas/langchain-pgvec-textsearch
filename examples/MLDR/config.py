import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_pgvec_textsearch import (
    SearchConfig,
    HNSWSearchConfig,
    BM25SearchConfig,
    HNSWIndexConfig,
    BM25IndexConfig,
    DistanceStrategy,
    RRFConfig,
    WSRConfig,
)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    data_dir: str

    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dim: int

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_ip: str
    postgres_port: str

    max_length: int = 8000


settings = Config()


# --- YAML experiment config loading ---

def load_experiment_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _parse_distance(s: str) -> DistanceStrategy:
    return DistanceStrategy(s)


def build_hnsw_index_config(cfg: dict) -> HNSWIndexConfig:
    return HNSWIndexConfig(
        m=cfg.get("m", 16),
        ef_construction=cfg.get("ef_construction", 64),
        distance_strategy=_parse_distance(cfg.get("distance_strategy", "COSINE_DISTANCE")),
    )


def build_bm25_index_config(cfg: dict, default_text_config: str) -> BM25IndexConfig:
    return BM25IndexConfig(
        text_config=cfg.get("text_config", default_text_config),
    )


def build_search_config(cfg: dict, default_text_config: str) -> SearchConfig:
    hnsw_cfg = cfg.get("hnsw", {})
    bm25_cfg = cfg.get("bm25", {})

    kwargs = dict(
        enable_dense=cfg.get("enable_dense", True),
        enable_sparse=cfg.get("enable_sparse", True),
        hnsw=HNSWSearchConfig(
            k=hnsw_cfg.get("k", 20),
            ef_search=hnsw_cfg.get("ef_search", 40),
            distance_strategy=_parse_distance(hnsw_cfg.get("distance_strategy", "COSINE_DISTANCE")),
        ),
        bm25=BM25SearchConfig(
            k=bm25_cfg.get("k", 20),
            text_config=bm25_cfg.get("text_config", default_text_config),
        ),
    )

    if "rrf" in cfg:
        kwargs["rrf"] = RRFConfig(**cfg["rrf"])
    if "wsr" in cfg:
        kwargs["wsr"] = WSRConfig(**cfg["wsr"])

    return SearchConfig(**kwargs)


def build_search_configs(index_cfg: dict, eval_cfg: dict) -> dict[str, SearchConfig]:
    text_config = index_cfg["dataset"]["text_config"]
    return {
        name: build_search_config(sc, text_config)
        for name, sc in eval_cfg["search_configs"].items()
    }
