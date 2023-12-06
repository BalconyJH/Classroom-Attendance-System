from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base

from src.utils.log import logger

Base = declarative_base()


class BaseModel(Base):
    """所有数据库模型的基类。"""
    __abstract__ = True

    @staticmethod
    @asynccontextmanager
    async def auto_commit(session: AsyncSession):
        """
        自动管理数据库事务的异步上下文管理器。
        在退出上下文时自动提交事务。如果发生异常，则回滚事务。

        参数:
            session: AsyncSession 实例，代表当前数据库会话。

        用法:
            ```python
            async with BaseModel.auto_commit(session):
                # 执行数据库操作
            ```
        """
        try:
            yield
            await session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Database Error: {e}")
            await session.rollback()
            logger.info("Rollback Transaction")

    @classmethod
    async def create(cls, db: AsyncSession, **kwargs):
        """
        创建并添加一个新实例到数据库。

        参数:
            db: AsyncSession 实例，代表当前数据库会话。
            **kwargs: 实例化模型所需的字段参数。

        返回:
            新创建的实例。

        用法:
            ```python
            user = await UserModel.create(session, name="Alice", email="alice@example.com")
            ```
        """
        instance = cls(**kwargs)
        async with cls.auto_commit(db):
            db.add(instance)
            await db.refresh(instance)
        return instance

    @classmethod
    async def read(cls, db: AsyncSession, _id: int):
        """
        根据 ID 从数据库读取一个实例。

        参数:
            db: AsyncSession 实例，代表当前数据库会话。
            _id: 要查询的实例 ID。

        返回:
            查询到的实例，如果不存在则为 None。

        用法:
            ```python
            user = await UserModel.read(session, user_id)
            ```
        """
        return await db.get(cls, _id)

    @classmethod
    async def read_all(cls, db: AsyncSession, **filters):
        """
        根据给定的筛选条件查询多个实例。

        参数:
            db: AsyncSession 实例，代表当前数据库会话。
            **filters: 筛选条件。

        返回:
            符合筛选条件的实例列表。

        用法:
            ```python
            users = await UserModel.read_all(session, name="Alice")
            ```
        """
        query = select(cls).filter_by(**filters)
        result = await db.execute(query)
        return result.scalars().all()

    async def update(self, db: AsyncSession, **kwargs):
        """
        更新实例的字段。

        参数:
            db: AsyncSession 实例，代表当前数据库会话。
            **kwargs: 需要更新的字段和值。

        返回:
            更新后的实例。

        用法:
            ```python
            user = await user.update(session, email="new_email@example.com")
            ```
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        async with self.auto_commit(db):
            pass
        return self

    async def delete(self, db: AsyncSession):
        """
        从数据库删除当前实例。

        参数:
            db: AsyncSession 实例，代表当前数据库会话。

        用法:
            ```python
            await user.delete(session)
            ```
        """
        async with self.auto_commit(db):
            await db.delete(self)

