from flask import session
from typing import Union

from utils.model import TeacherSession, StudentSession


class SessionManager:
    @staticmethod
    def load_session_data() -> Union[StudentSession, TeacherSession, None]:
        """从 Flask session 中加载用户会话数据, 返回对应的 Pydantic 模型。"""
        role = session.get("role")
        if role == "student":
            return StudentSession(**session)
        elif role == "teacher":
            return TeacherSession(**session)
        return None

    @staticmethod
    def update_session_data(data: Union[StudentSession, TeacherSession]) -> None:
        """更新 Flask session 数据。"""
        session.update(data.model_dump(exclude_none=True))

    @staticmethod
    def clear_session() -> None:
        """清除 Flask session 数据。"""
        session.clear()
