from contextlib import asynccontextmanager

from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.config import config

from .db_model import Base

DATABASE_URL = config.database_url
# if config.log_level != "INFO":
#     echo = True
# else:
echo = False
# 创建异步引擎
engine = create_async_engine(DATABASE_URL, echo=echo, future=True)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def test_db_connection() -> bool:
    try:
        async with engine.begin():
            return True
    except SQLAlchemyError:
        return False


@asynccontextmanager
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
