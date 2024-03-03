from pathlib import Path
from typing import Optional

from flask import session
from sqlalchemy import extract
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.database.models import Attendance, Course, Faces, StudentCourse, Student, Teacher


async def get_student_by_id(student_id: str) -> Student:
    """根据学生ID获取学生实例"""
    return Student.query.get(student_id)


async def get_recent_attendances(student_id: str, limit: int = 5) -> list[Attendance]:
    """获取最近的考勤记录"""
    return Attendance.query.filter(Attendance.s_id == student_id).order_by(Attendance.time.desc()).limit(limit).all()


async def get_attendance_records(student_id: str, limit: int = 5) -> dict[Attendance, Course]:
    """获取考勤记录及对应的课程"""
    attendances = await get_recent_attendances(student_id, limit)
    records = {}
    for attendance in attendances:
        course = Course.query.get(attendance.c_id)
        records[attendance] = course
    return records


async def count_attendance_by_result(student_id: str, month: int, year: int, result: str) -> int:
    """根据结果计算考勤次数"""
    return Attendance.query.filter(
        Attendance.s_id == student_id,
        extract("month", Attendance.time) == month,
        extract("year", Attendance.time) == year,
        Attendance.result == result,
    ).count()


async def get_monthly_attendance_summary(student_id: str, month: int, year: int) -> dict[str, int]:
    """获取每月的考勤摘要"""
    return {
        "leave": await count_attendance_by_result(student_id, month, year, "请假"),
        "late": await count_attendance_by_result(student_id, month, year, "迟到"),
        "absent": await count_attendance_by_result(student_id, month, year, "缺勤"),
        "checked": await count_attendance_by_result(student_id, month, year, "已签到"),
    }


def pre_work_mkdir(path_photos_from_camera):
    path = Path(path_photos_from_camera)
    if not path.is_dir():
        path.mkdir()


async def update_database_with_features(features):
    student_face = Faces.query.filter(Faces.s_id == session["id"]).first()
    if student_face:
        student_face.feature = features
    else:
        face = Faces(s_id=session["id"], feature=features)
        db.session.add(face)
    db.session.commit()
    await update_student_flag()


async def update_student_flag():
    student_record = Student.query.filter(Student.s_id == session["id"]).first()
    if student_record:
        student_record.flag = 0
        db.session.commit()


async def get_courses_by_student_id(s_id: str) -> list[Course]:
    """根据学生ID获取其所有课程。"""
    return (
        Course.query.join(StudentCourse, Course.c_id == StudentCourse.c_id)
        .filter(StudentCourse.s_id == s_id)
        .order_by(Course.c_id)
        .all()
    )


async def get_records_by_course_and_time(
    s_id: str, c_id: Optional[str] = None, time: Optional[str] = None
) -> dict[Course, list[Attendance]]:
    """根据课程ID和时间获取指定学生的考勤记录。"""
    records_dict = {}
    courses = (
        Course.query.join(StudentCourse, Course.c_id == StudentCourse.c_id).filter(StudentCourse.s_id == s_id)
        if not c_id
        else [Course.query.filter_by(c_id=c_id).first()]
    )

    for course in courses:
        query = Attendance.query.filter_by(s_id=s_id, c_id=course.c_id)
        if time:
            query = query.filter(Attendance.time.like(f"{time}%"))
        records = query.all()
        records_dict[course] = records

    return records_dict


async def add_student_course(s_id: str, c_id: str) -> bool:
    """添加学生选课记录"""
    try:
        sc = StudentCourse(s_id=s_id, c_id=c_id)
        db.session.add(sc)
        db.session.commit()
        return True
    except SQLAlchemyError:
        db.session.rollback()
        return False


async def get_student_courses(s_id: str) -> list[StudentCourse]:
    """获取学生已选课程"""
    return StudentCourse.query.filter_by(s_id=s_id).all()


async def get_available_courses(s_id: str) -> dict[Course, Teacher]:
    """获取学生未选的可选课程及其教师信息"""
    result_dict = {}
    # 获取学生已选的所有课程ID
    enrolled_course_ids = [sc.c_id for sc in await get_student_courses(s_id)]
    # 查询未选且可选的课程
    available_courses = Course.query.filter(Course.c_id.notin_(enrolled_course_ids), Course.flag == "可选课").all()
    for course in available_courses:
        teacher = Teacher.query.filter_by(t_id=course.t_id).first()
        if teacher:
            result_dict[course] = teacher
    return result_dict


async def delete_student_course(sid: str, cid: str) -> None:
    """删除学生的选课记录"""
    sc = StudentCourse.query.filter_by(c_id=cid, s_id=sid).first()
    if sc:
        db.session.delete(sc)
        db.session.commit()


async def get_selectable_courses_and_teachers(sid: str) -> dict[Course, Teacher]:
    """获取学生可退选的课程及其任课教师"""
    selected_courses = await get_student_courses(sid)
    courses_id_list = [sc.c_id for sc in selected_courses]
    courses_and_teachers = {}
    if courses_id_list:
        selectable_courses = Course.query.filter(Course.c_id.in_(courses_id_list), Course.flag == "可选课").all()
        for course in selectable_courses:
            teacher = Teacher.query.filter_by(t_id=course.t_id).first()
            courses_and_teachers[course] = teacher
    return courses_and_teachers


async def update_user_password(user_type: str, user_id: str, old_password: str, new_password: str) -> bool:
    user = None
    if user_type == "student":
        user = Student.query.filter_by(s_id=user_id).first()
        password_field = "s_password"
    elif user_type == "teacher":
        user = Teacher.query.filter_by(t_id=user_id).first()
        password_field = "t_password"
    else:
        return False

    if user is None:
        return False

    if getattr(user, password_field) == old_password:
        setattr(user, password_field, new_password)
        db.session.commit()
        return True
    else:
        return False
