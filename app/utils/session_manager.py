from typing import Union

from flask import session

from app.utils.model import StudentSession, TeacherSession


class SessionManager:
    @staticmethod
    def load_session_data() -> Union[StudentSession, TeacherSession, None]:
        """
        从 Flask session 中加载会话数据。
        :return: None 或 StudentSession 或 TeacherSession 对象。
        """
        role = session.get("role")
        if role == "student":
            return StudentSession(**session)
        elif role == "teacher":
            return TeacherSession(**session)
        return None

    @staticmethod
    def update_session_data(data: Union[StudentSession, TeacherSession]) -> None:
        """
        更新 Flask session 数据。
        :param data: StudentSession 或 TeacherSession 对象。
        :return: None
        """
        session.update(data.model_dump(exclude_none=True))

    @staticmethod
    def clear_session() -> None:
        """
        清空 Flask session。
        :return: None
        """
        session.clear()
