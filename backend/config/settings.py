from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "ML Assistant API"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Database (defaults to SQLite so the platform runs fully offline/locally for testing
    # without Postgres or any cloud dependency; set DATABASE_URL to a Postgres DSN to use one).
    database_url: str = Field(
        default="sqlite+aiosqlite:///./automl_app.db",
        alias="DATABASE_URL",
    )
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", alias="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS",
    )

    # File storage
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=100, alias="MAX_FILE_SIZE_MB")

    # ML
    experiment_timeout_seconds: int = Field(default=300, alias="EXPERIMENT_TIMEOUT_SECONDS")
    max_concurrent_experiments: int = Field(default=3, alias="MAX_CONCURRENT_EXPERIMENTS")

    # Redis (only used by the optional Celery path; the AutoML engine runs locally)
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    # ===== Interactive AutoML & Multi-Agent Engine =====
    # Working/artifact directories
    automl_output_dir: str = Field(default="./automl_outputs", alias="AUTOML_OUTPUT_DIR")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")

    # Central Single-Source-Of-Truth dataset store (DuckDB preferred, SQLite fallback)
    central_store_path: str = Field(default="./automl_outputs/central_store.duckdb", alias="CENTRAL_STORE_PATH")
    central_store_in_memory: bool = Field(default=False, alias="CENTRAL_STORE_IN_MEMORY")

    # MLflow (local file tracking by default -> no cloud)
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="interactive_automl", alias="MLFLOW_EXPERIMENT_NAME")

    # LLM provider: "local" (no keys needed), "openai", "anthropic"
    llm_provider: str = Field(default="local", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-haiku-latest", alias="ANTHROPIC_MODEL")

    # Paper research & knowledge retrieval (HTTP-based, no MCP daemon required)
    semantic_scholar_api_key: str = Field(default="", alias="SEMANTIC_SCHOLAR_API_KEY")
    arxiv_mcp_server_url: str = Field(default="http://localhost:8000/mcp/arxiv", alias="ARXIV_MCP_SERVER_URL")
    research_request_timeout: int = Field(default=12, alias="RESEARCH_REQUEST_TIMEOUT")
    enable_offline_paper_bank: bool = Field(default=True, alias="ENABLE_OFFLINE_PAPER_BANK")

    # AutoML run control
    automl_max_iterations: int = Field(default=12, alias="AUTOML_MAX_ITERATIONS")
    automl_default_max_rows: int = Field(default=50_000, alias="AUTOML_DEFAULT_MAX_ROWS")
    automl_default_metric: str = Field(default="accuracy", alias="AUTOML_DEFAULT_METRIC")
    automl_default_target_value: float = Field(default=0.85, alias="AUTOML_DEFAULT_TARGET_VALUE")
    automl_metric_direction: str = Field(default="maximize", alias="AUTOML_METRIC_DIRECTION")
    automl_feature_engineering_every_n: int = Field(default=2, alias="AUTOML_FEATURE_ENGINEERING_EVERY_N")
    automl_test_size: float = Field(default=0.2, alias="AUTOML_TEST_SIZE")
    automl_model_time_budget: int = Field(default=45, alias="AUTOML_MODEL_TIME_BUDGET")

    # Kaggle connector (optional; only used when credentials supplied)
    kaggle_username: str = Field(default="", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field(default="", alias="KAGGLE_KEY")
    huggingface_token: str = Field(default="", alias="HUGGINGFACE_TOKEN")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
