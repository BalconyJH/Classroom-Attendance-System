from pydantic import BaseModel

from app import db


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(320), unique=True)
    password = db.Column(db.String(32), nullable=False)

    def __repr__(self):
        return "<User %r>" % self.username


class Admin(db.Model):
    __tablename__ = "admins"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(320), unique=True)
    password = db.Column(db.String(32), nullable=False)

    def __repr__(self):
        return "<User %r>" % self.username


class Student(db.Model):
    __tablename__ = "students"
    s_id = db.Column(db.String(13), primary_key=True)
    s_name = db.Column(db.String(80), nullable=False)
    s_password = db.Column(db.String(32), nullable=False)
    flag = db.Column(db.Integer, default=1)
    before = db.Column(db.DateTime)

    def __repr__(self):
        return f"<Student {self.s_id!r},{self.s_name!r}>"


class Teacher(db.Model):
    __tablename__ = "teachers"
    t_id = db.Column(db.String(8), primary_key=True)
    t_name = db.Column(db.String(80), nullable=False)
    t_password = db.Column(db.String(32), nullable=False)
    before = db.Column(db.DateTime)

    def __repr__(self):
        return f"<Teacher {self.t_id!r},{self.t_name!r}>"


class Faces(db.Model):
    __tablename__ = "student_faces"
    s_id = db.Column(db.String(13), primary_key=True)
    feature = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Faces {self.s_id!r}>"


class Course(db.Model):
    __tablename__ = "courses"
    c_id = db.Column(db.String(6), primary_key=True)
    t_id = db.Column(db.String(8), db.ForeignKey("teachers.t_id"), nullable=False)
    c_name = db.Column(db.String(100), nullable=False)
    times = db.Column(db.Text, default="0000-00-00 00:00")
    flag = db.Column(db.String(50), default="不可选课")

    def __repr__(self):
        return f"<Course {self.c_id!r},{self.t_id!r},{self.c_name!r}>"


class StudentCourse(db.Model):
    __tablename__ = "student_course"
    s_id = db.Column(db.String(13), db.ForeignKey("students.s_id"), primary_key=True)
    c_id = db.Column(db.String(100), db.ForeignKey("courses.c_id"), primary_key=True)

    def __repr__(self):
        return f"<StudentCourse {self.s_id!r},{self.c_id!r}> "


class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    s_id = db.Column(db.String(13), db.ForeignKey("students.s_id"))
    c_id = db.Column(db.String(100), db.ForeignKey("courses.c_id"))
    time = db.Column(db.DateTime)
    result = db.Column(db.String(10), nullable=False)

    def __repr__(self):
        return f"<Attendance {self.s_id!r},{self.c_id!r},{self.time!r},{self.result!r}>"


class TimeID(BaseModel):
    id: int
    time: str

    def __hash__(self):
        return hash((self.id, self.time))

    def __eq__(self, other):
        return isinstance(other, TimeID) and self.id == other.id and self.time == other.time


class ChooseCourse(db.Model):
    __tablename___ = "choose_course"
    c_id = db.Column(db.String(6), primary_key=True)
    t_id = db.Column(db.String(8), nullable=False)
    c_name = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<Course {self.c_id!r},{self.t_id!r},{self.c_name!r}>"
