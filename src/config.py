from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    typed .env variables
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openrouter_api_key: str = Field(
        default="",
        validation_alias="OPENROUTER_API_KEY",
        description="OpenRouter API Key",
    )
    llm_model: str = Field(default="qwen/qwen3-235b-a22b:free")

    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    collection_name: str = Field(default="rag_documents")

    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)

    top_k: int = Field(default=5, ge=1, le=20)


settings = Settings()
