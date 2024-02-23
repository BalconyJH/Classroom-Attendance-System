from pydantic import BaseModel, Field
from typing import Optional


class UserSessionBase(BaseModel):
    username: str
    id: str
    name: str
    role: str
    time: Optional[str] = None


class StudentSession(UserSessionBase):
    num: int = Field(0, alias="num")
    flag: bool = Field(True, alias="flag")


class TeacherSession(UserSessionBase):
    attend: list[str] = Field(default_factory=list, alias="attend")