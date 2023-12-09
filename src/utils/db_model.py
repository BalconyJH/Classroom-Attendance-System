from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Boolean, Integer, DateTime, select

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
            await db.commit()  # 确保首先提交事务
            await db.refresh(instance)  # 提交后刷新实例
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


class User(BaseModel):
    """用户模型。"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    # attendances = relationship("Attendance", back_populates="user", cascade="all, delete-orphan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    async def find_user(cls, session: AsyncSession, **kwargs):
        """
        根据给定的筛选条件查询一个用户。
        :param session:  AsyncSession 实例，代表当前数据库会话。
        :param kwargs:  筛选条件。
        :return:  符合筛选条件的用户。

        用法:
            ```python
            user = await UserModel.find_user(session, email="111@gmail.com")
            ```
        """
        return await cls.read_all(session, **kwargs)

    @classmethod
    async def create_admin(cls, session: AsyncSession, **kwargs):
        """
        创建并添加一个新管理员到数据库。
        :param session: AsyncSession 实例，代表当前数据库会话。
        :param kwargs: 实例化模型所需的字段参数。
        :return: 创建的管理员实例。
        """
        kwargs["is_admin"] = True
        instance = cls(**kwargs)
        async with cls.auto_commit(session):
            session.add(instance)
            await session.refresh(instance)
        return instance
