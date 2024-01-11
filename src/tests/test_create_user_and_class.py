import pytest

from src.database.db_model import User


@pytest.mark.asyncio
async def test_create_user_and_class(async_session):
    # 使用 async_session 实例进行数据库操作
    async for session in async_session:
        # 这里使用 session 进行数据库操作
        await User.create(session, name="John Doe", student_id="123", password="password")
        # 其他需要的测试或断言
