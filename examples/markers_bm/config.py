from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown vars in env/.env
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
