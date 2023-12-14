
import pytest

from src.database.db_model import User


@pytest.mark.asyncio
async def test_create_user_and_class(async_session):
    # 创建一个新用户
    new_user = await User.create(async_session, name="John Doe", student_id="123", password="password")
    assert new_user is not None

    found_user = await User.read(async_session, new_user.id)
    assert found_user is not None

    # 清理：删除测试数据（可选）
    await new_user.delete(async_session)
