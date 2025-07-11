from pydantic_settings import BaseSettings, SettingsConfigDict
# from dotenv import load_dotenv
# import os
from pathlib import Path


# --- Environment Variable Loading ---
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / 'app' / '.env'

if not dotenv_path.exists():
    raise FileNotFoundError(f".env file not found at {dotenv_path}")

# load_dotenv(dotenv_path)
# print(os.getenv("POLLY_ACCESS_KEY_ID"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=dotenv_path, extra="ignore", env_file_encoding="utf-8")

    OPENAI_API_KEY: str
    WHATSAPP_API_URL: str
    WHATSAPP_API_AUTHORIZATION: str

    DEFAULT_ADMIN: str
    OFFICE_LONGITUDE: str
    OFFICE_LATITUDE: str
    OFFICE_ADDRESS: str

    QDRANT_URL: str
    QDRANT_API_KEY: str

    ASSEMBLYAI_API_KEY: str

    POLLY_ACCESS_KEY_ID: str
    POLLY_SECRET_ACCESS_KEY: str
    POLLY_REGION_NAME: str

    DATABASE_URL: str
    NON_ASYNC_DATABASE_URL: str

    S3_BUCKET_URL: str
    S3_BUCKET_ACCESS_KEY_ID: str
    S3_BUCKET_SECRET_ACCESS_KEY: str
    S3_REGION_NAME: str

    TOGETHERAI_API_KEY: str

    QDRANT_URL: str
    QDRANT_API_KEY: str

    MONGODBATLAS_URI: str

    TAVILY_API_KEY: str


settings = Settings()
print(settings.NON_ASYNC_DATABASE_URL)
