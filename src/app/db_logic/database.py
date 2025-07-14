from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
import asyncio
from sqlalchemy.orm import sessionmaker
from typing import Optional, List, Dict, Callable, Any
from app.settings import settings
from sqlalchemy.exc import OperationalError, IntegrityError, StatementError, TimeoutError
from tenacity import retry, wait_exponential, stop_after_attempt, before_log, after_log, retry_if_exception_type
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DATABASE_URL = settings.DATABASE_URL

RETRIABLE_DB_EXCEPTIONS = (OperationalError, TimeoutError, StatementError)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRIABLE_DB_EXCEPTIONS),
    before_sleep=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARNING),
    reraise=True
)
async def execute_with_retry(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    """Execute an async function with retry logic for database operations."""
    if not asyncio.iscoroutinefunction(async_func):
        raise ValueError(
            f"Function {async_func.__name__} must be an async function")
    return await async_func(*args, **kwargs)


# Create an asynchronous engine
engine = create_async_engine(DATABASE_URL)

# Declarative base for defining models
Base = declarative_base()

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
