import pytest

from src.database.db_model import User


@pytest.mark.asyncio
async def test_create_user_and_class(async_session):
    user = User(name="John Doe", student_id="123", password="password")
    async_session.add(user)
    await async_session.commit()
