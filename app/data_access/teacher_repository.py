import time
from typing import Union, Optional

from app import db
from app.database.models import Course, Student, Teacher, Attendance, StudentCourse


async def get_courses_by_teacher_id(teacher_id: str) -> list[Course]:
    """根据教师ID获取其所有课程。

    参数:
        teacher_id (int): 教师的ID。

    返回:
        List[Course]: 包含教师所有课程的列表。
    """
    courses = Course.query.filter_by(t_id=teacher_id).all()
    return courses


async def get_student_count_by_course_id(course_id: str) -> int:
    """根据课程ID获取选修该课程的学生数量。

    参数:
        course_id (str): 课程的ID。

    返回:
        int: 选修该课程的学生数量。
    """
    count = StudentCourse.query.filter_by(c_id=course_id).count()
    return count


def update_course_times(cid: str) -> Union[str, None]:
    """更新课程的考勤次数"""
    course = Course.query.filter_by(c_id=cid).first()
    if course:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        course.times = f"{course.times}/{now}"
        db.session.commit()
        return now
    return None


def initialize_attendance_records(cid: str, now: str) -> None:
    """初始化考勤记录, 标记所有学生为未签到"""
    the_course_students = StudentCourse.query.filter_by(c_id=cid)
    all_students_attend = [Attendance(s_id=sc.s_id, c_id=cid, time=now, result="缺勤") for sc in the_course_students]
    db.session.add_all(all_students_attend)
    db.session.commit()


def sid_if_exist(sid):
    num = Student.query.filter(Student.s_id.in_(sid)).count()
    return num


def cid_if_exist(cid):
    num = Course.query.filter(Course.c_id.in_(cid)).count()
    return num


def tid_if_exist(tid):
    num = Teacher.query.filter(Teacher.t_id.in_(tid)).count()
    return num


def mark_student_as_deleted(sid):
    """标记学生为已删除状态, 并提交数据库更改。"""
    student = Student.query.filter(Student.s_id == sid).first()
    if student:
        student.flag = 1  # 假设flag=1表示学生已被删除
        db.session.commit()


async def update_attendance(course_id: str, student_id: str, class_time: str, result: str) -> Optional[Attendance]:
    """
    更新学生的出勤记录。

    参数:
        course_id (str): 课程ID。
        student_id (str): 学生ID。
        time (datetime): 上课时间。
        result (str): 出勤结果。

    返回:
        Optional[Attendance]: 更新后的出勤记录对象, 如果找不到则返回None。
    """
    one_attend = Attendance.query.filter_by(c_id=course_id, s_id=student_id, time=class_time).first()
    if one_attend:
        one_attend.result = result
        db.session.commit()
        return one_attend
    return None


async def update_attendance_records(all_sid: list[str], cid: str, all_time: str) -> None:
    """
    更新考勤记录为"已签到"。

    :param all_sid: 学生ID列表。
    :param cid: 课程ID。
    :param all_time: 考勤时间。
    :return: None
    """
    Attendance.query.filter(
        Attendance.time == all_time,
        Attendance.c_id == cid,
        Attendance.s_id.in_(all_sid),
    ).update({"result": "已签到"})
    db.session.commit()
