import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.database.db_model import Base


@pytest.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    async for engine in async_engine:
        async_session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session_factory() as session:
            yield session

