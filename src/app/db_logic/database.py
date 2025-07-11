from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
# import asyncio
from sqlalchemy.orm import sessionmaker
from app.settings import settings

DATABASE_URL = settings.DATABASE_URL

# Create an asynchronous engine
engine = create_async_engine(DATABASE_URL)

# Declarative base for defining models
Base = declarative_base()

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
