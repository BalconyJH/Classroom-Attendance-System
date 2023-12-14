import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.database.db_model import Base


# 创建异步测试数据库引擎
@pytest.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


# 异步会话工厂
@pytest.fixture
async def async_session(async_engine, async_session):
    async_session_factory = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_factory() as async_session:
        yield async_session
