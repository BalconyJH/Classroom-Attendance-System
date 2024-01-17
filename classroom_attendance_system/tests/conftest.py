import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from utils import logger
from database.db_model import Base

# @pytest.fixture
# async def async_engine():
#     """
#     创建一个内存中的 sqlite 数据库引擎。
#     引擎会在测试结束后自动销毁。
#     引擎启动时会自动创建所有模型对应的表。
#     :return: AsyncEngine 实例。
#     """
#     engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#         logger.info("Create all tables")
#     yield engine
#     await engine.dispose()
#
#
# @pytest.fixture
# async def async_session(async_engine):
#     async for engine in async_engine:
#         async_session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
#         async with async_session_factory() as session:
#             yield session


@pytest.fixture
async def async_session(request):
    async_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(async_engine) as session:
        yield session

    await async_engine.dispose()
    logger.info("Dispose engine")
