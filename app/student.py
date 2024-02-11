import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Blueprint, flash, request, session, url_for, redirect, render_template
from flask import FlaskException
from sqlalchemy import extract
from sqlalchemy.exc import SQLAlchemyError

from app import db, app
from config import config
from .face_feature_processor import FaceFeatureProcessor
from .models import Faces, Course, Student, Teacher, Attendance, StudentCourse

student = Blueprint("student", __name__, static_folder="static")


def save_image(data: bytes, path: Path) -> None:
    """
    保存图片
    :param data: 图片数据
    :param path: 保存路径
    :return: None
    """
    with open(path, "wb") as file:
        file.write(data)


def get_student_by_id(student_id: str) -> Student:
    """根据学生ID获取学生实例"""
    return Student.query.get(student_id)


def get_recent_attendances(student_id: str, limit: int = 5) -> list[Attendance]:
    """获取最近的考勤记录"""
    return Attendance.query.filter(Attendance.s_id == student_id).order_by(Attendance.time.desc()).limit(limit).all()


def get_attendance_records(student_id: str, limit: int = 5) -> dict[Attendance, Course]:
    """获取考勤记录及对应的课程"""
    attendances = get_recent_attendances(student_id, limit)
    records = {}
    for attendance in attendances:
        course = Course.query.get(attendance.c_id)
        records[attendance] = course
    return records


def count_attendance_by_result(student_id: str, month: int, year: int, result: str) -> int:
    """根据结果计算考勤次数"""
    return Attendance.query.filter(
        Attendance.s_id == student_id,
        extract("month", Attendance.time) == month,
        extract("year", Attendance.time) == year,
        Attendance.result == result,
    ).count()


def get_monthly_attendance_summary(student_id: str, month: int, year: int) -> dict[str, int]:
    """获取每月的考勤摘要"""
    return {
        "leave": count_attendance_by_result(student_id, month, year, "请假"),
        "late": count_attendance_by_result(student_id, month, year, "迟到"),
        "absent": count_attendance_by_result(student_id, month, year, "缺勤"),
        "checked": count_attendance_by_result(student_id, month, year, "已签到"),
    }


@student.route("/home")
def home():
    student_id = session["id"]
    student_instance = get_student_by_id(student_id)
    session["flag"] = student_instance.flag
    records = get_attendance_records(student_id)
    month = datetime.now().month
    year = datetime.now().year
    num = get_monthly_attendance_summary(student_id, month, year)

    return render_template(
        "student/student_home.html",
        flag=session["flag"],
        before=session["time"],
        records=records,
        name=session["name"],
        num=num,
    )


def pre_work_mkdir(path_photos_from_camera):
    path = Path(path_photos_from_camera)
    if not path.is_dir():
        path.mkdir()


@student.route("/get_faces", methods=["GET", "POST"])
def get_faces():
    if request.method == "POST":
        imgdata = base64.b64decode(request.form.get("face"))
        path = Path(config.cache_path / "dataset" / session["id"])
        face_register = FaceFeatureProcessor(app.logger)
        if session["num"] == 0:
            pre_work_mkdir(path)
        if session["num"] == 5:
            session["num"] = 0
        session["num"] += 1
        current_face_path = path / f"{session['num']}.jpg"
        save_image(imgdata, current_face_path)
        flag = face_register.process_single_face_image(str(current_face_path))
        if flag != "right":
            session["num"] -= 1
        return {"result": flag, "code": session["num"]}
    return render_template("student/get_faces.html")


def extract_and_process_features():
    path = os.path.join(config.cache_path, "dataset", session["id"])
    average_face_features = FaceFeatureProcessor(app.logger).calculate_average_face_features(str(path))
    features = ",".join(str(feature) for feature in average_face_features)
    app.logger.info(" >> 特征均值 / The mean of features:", list(average_face_features), "\n")
    return features


def update_database_with_features(features):
    student_face = Faces.query.filter(Faces.s_id == session["id"]).first()
    if student_face:
        student_face.feature = features
    else:
        face = Faces(s_id=session["id"], feature=features)
        db.session.add(face)
    db.session.commit()
    update_student_flag()


def update_student_flag():
    student_record = Student.query.filter(Student.s_id == session["id"]).first()
    if student_record:
        student_record.flag = 0
        db.session.commit()


@student.route("/upload_faces", methods=["POST"])
def upload_faces():
    try:
        # 提取特征并处理数据
        features = extract_and_process_features()
        # 更新数据库
        update_database_with_features(features)
        # 设置成功消息并重定向
        flash("提交成功! ")
        return redirect(url_for("student.home"))
    except FlaskException as e:
        app.logger.debug("Error:", e)
        flash("提交不合格照片, 请拍摄合格后再重试")
        return redirect(url_for("student.home"))


@student.route("/my_faces")
def my_faces():
    current_face_path = Path(config.cache_path) / "dataset" / session["id"]

    face_path = Path("static/caches/dataset") / session["id"]

    photos_list = list(current_face_path.glob("*.jpg"))
    num = len(photos_list)

    paths = [str(face_path / f"{i + 1}.jpg") for i in range(num)]

    return render_template("student/my_faces.html", face_paths=paths)


def get_courses_by_student_id(s_id: str) -> list[Course]:
    """根据学生ID获取其所有课程。"""
    return (
        Course.query.join(StudentCourse, Course.c_id == StudentCourse.c_id)
        .filter(StudentCourse.s_id == s_id)
        .order_by(Course.c_id)
        .all()
    )


def get_records_by_course_and_time(
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


@student.route("/my_records", methods=["GET", "POST"])
def my_records():
    sid = session["id"]
    courses = get_courses_by_student_id(sid)
    if request.method == "POST":
        cid = str(request.form.get("course_id", ""))
        time = str(request.form.get("time", ""))
        records_dict = get_records_by_course_and_time(sid, cid if cid else None, time if time else None)
    else:
        records_dict = get_records_by_course_and_time(sid)
    return render_template("student/my_records.html", dict=records_dict, courses=courses)


def add_student_course(s_id: str, c_id: str) -> bool:
    """添加学生选课记录"""
    try:
        sc = StudentCourse(s_id=s_id, c_id=c_id)
        db.session.add(sc)
        db.session.commit()
        return True
    except SQLAlchemyError:
        db.session.rollback()
        return False


def get_student_courses(s_id: str) -> list[StudentCourse]:
    """获取学生已选课程"""
    return StudentCourse.query.filter_by(s_id=s_id).all()


def get_available_courses(s_id: str) -> dict[Course, Teacher]:
    """获取学生未选的可选课程及其教师信息"""
    result_dict = {}
    # 获取学生已选的所有课程ID
    enrolled_course_ids = [sc.c_id for sc in get_student_courses(s_id)]
    # 查询未选且可选的课程
    available_courses = Course.query.filter(Course.c_id.notin_(enrolled_course_ids), Course.flag == "可选课").all()
    for course in available_courses:
        teacher = Teacher.query.filter_by(t_id=course.t_id).first()
        if teacher:
            result_dict[course] = teacher
    return result_dict


@student.route("/choose_course", methods=["GET", "POST"])
def choose_course():
    try:
        sid = session["id"]
        if request.method == "POST":
            cid = request.form.get("cid")
            if not add_student_course(sid, cid):
                flash("选课失败, 请重试")
                return redirect(url_for("student.choose_course"))

        available_courses = get_available_courses(sid)
        return render_template("student/choose_course.html", dict=available_courses)
    except FlaskException:
        flash("未知错误操作")
        return redirect(url_for("student.home"))


def delete_student_course(sid: str, cid: str) -> None:
    """删除学生的选课记录"""
    sc = StudentCourse.query.filter_by(c_id=cid, s_id=sid).first()
    if sc:
        db.session.delete(sc)
        db.session.commit()


def get_selectable_courses_and_teachers(sid: str) -> dict[Course, Teacher]:
    """获取学生可退选的课程及其任课教师"""
    selected_courses = get_student_courses(sid)
    cids = [sc.c_id for sc in selected_courses]
    courses_and_teachers = {}
    if cids:
        selectable_courses = Course.query.filter(Course.c_id.in_(cids), Course.flag == "可选课").all()
        for course in selectable_courses:
            teacher = Teacher.query.filter_by(t_id=course.t_id).first()
            courses_and_teachers[course] = teacher
    return courses_and_teachers


@student.route("/drop_course", methods=["GET", "POST"])
def drop_course():
    try:
        sid = session["id"]
        if request.method == "POST":
            cid = request.form.get("cid")
            delete_student_course(sid, cid)  # 删除选课记录

        selectable_courses = get_selectable_courses_and_teachers(sid)  # 获取可选课程及其教师

        return render_template("student/drop_course.html", dict=selectable_courses)
    except FlaskException:
        flash("未知错误")
        return redirect(url_for("student.home"))


def update_user_password(user_type: str, user_id: str, old_password: str, new_password: str) -> bool:
    """更新用户密码"""
    if user_type == "student":
        user = Student.query.filter_by(s_id=user_id).first()
    elif user_type == "teacher":
        user = Teacher.query.filter_by(t_id=user_id).first()
    else:
        flash("未知用户类型")
        return False

    if user is None:
        flash("用户不存在")
        return False

    if user.s_password == old_password or user.t_password == old_password:
        user.s_password = new_password if user_type == "student" else None
        user.t_password = new_password if user_type == "teacher" else None
        db.session.commit()
        flash("密码修改成功!")
        return True
    else:
        flash("旧密码错误, 请重试")
        return False


@student.route("/update_password", methods=["GET", "POST"])
def update_password():
    user_id = session["id"]
    if request.method == "POST":
        old = request.form.get("old")
        new = request.form.get("new")
        update_user_password("student", user_id, old, new)
    return render_template(
        "student/update_password.html", student=Student.query.filter(Student.s_id == user_id).first()
    )
