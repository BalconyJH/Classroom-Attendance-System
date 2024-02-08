import os
import base64
from pathlib import Path
from datetime import datetime

from config import config
from sqlalchemy import extract
from flask import Blueprint, flash, request, session, url_for, redirect, render_template

from app import db, app

from .face_feature_processor import FaceFeatureProcessor
from .models import Faces, Course, Student, Teacher, Attendance, StudentCourse

student = Blueprint("student", __name__, static_folder="static")


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
    # 新建文件夹 / Create folders to save faces images and csv
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)


@student.route("/get_faces", methods=["GET", "POST"])
def get_faces():
    if request.method == "POST":
        imgdata = request.form.get("face")
        imgdata = base64.b64decode(imgdata)
        path = Path(config.cache_path / "dataset" / session["id"])
        face_register = FaceFeatureProcessor(app.logger)
        if session["num"] == 0:
            pre_work_mkdir(path)
        if session["num"] == 5:
            session["num"] = 0
        session["num"] += 1
        current_face_path = path / f"{session['num']}.jpg"
        with open(current_face_path, "wb") as f:
            f.write(imgdata)
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
    except Exception as e:
        app.logger.debug("Error:", e)
        flash("提交不合格照片, 请拍摄合格后再重试")
        return redirect(url_for("student.home"))


@student.route("/my_faces")
def my_faces():
    current_face_path = "app/static/caches/dataset/" + session["id"] + "/"
    face_path = "static/caches/dataset/" + session["id"] + "/"
    photos_list = os.listdir(current_face_path)
    num = len(photos_list)
    paths = []
    for i in range(num):
        path = face_path + str(i + 1) + ".jpg"
        paths.append(path)
    return render_template("student/my_faces.html", face_paths=paths)


@student.route("/my_records", methods=["GET", "POST"])
def my_records():
    sid = session["id"]
    dict = {}
    if request.method == "POST":
        cid = str(request.form.get("course_id"))
        time = str(request.form.get("time"))
        if cid != "" and time != "":
            course = Course.query.filter(Course.c_id == cid).first()
            one_course_records = (
                db.session.query(Attendance)
                .filter(
                    Attendance.s_id == sid,
                    Attendance.c_id == cid,
                    Attendance.time.like(time + "%"),
                )
                .all()
            )
            dict[course] = one_course_records
            courses = (
                db.session.query(Course).join(StudentCourse).filter(StudentCourse.s_id == sid).order_by("c_id").all()
            )
            return render_template("student/my_records.html", dict=dict, courses=courses)
        elif cid != "" and time == "":
            course = Course.query.filter(Course.c_id == cid).first()
            one_course_records = (
                db.session.query(Attendance).filter(Attendance.s_id == sid, Attendance.c_id == cid).all()
            )
            dict[course] = one_course_records
            courses = (
                db.session.query(Course).join(StudentCourse).filter(StudentCourse.s_id == sid).order_by("c_id").all()
            )
            return render_template("student/my_records.html", dict=dict, courses=courses)
        elif cid == "" and time != "":
            courses = (
                db.session.query(Course).join(StudentCourse).filter(StudentCourse.s_id == sid).order_by("c_id").all()
            )
            for course in courses:
                one_course_records = (
                    db.session.query(Attendance)
                    .filter(
                        Attendance.s_id == sid,
                        Attendance.c_id == course.c_id,
                        Attendance.time.like(time + "%"),
                    )
                    .order_by("c_id")
                    .all()
                )
                dict[course] = one_course_records
            courses = (
                db.session.query(Course).join(StudentCourse).filter(StudentCourse.s_id == sid).order_by("c_id").all()
            )
            return render_template("student/my_records.html", dict=dict, courses=courses)
        else:  # cid =='' and time ==''
            pass
    # all_course_record = []
    courses = db.session.query(Course).join(StudentCourse).filter(StudentCourse.s_id == sid).order_by("c_id").all()
    # print(courses)
    for course in courses:
        one_course_records = (
            db.session.query(Attendance)
            .filter(Attendance.s_id == sid, Attendance.c_id == course.c_id)
            .order_by("c_id")
            .all()
        )
        # all_course_record.append(one_course_records)
        dict[course] = one_course_records
    # print(dict)
    return render_template("student/my_records.html", dict=dict, courses=courses)


@student.route("/choose_course", methods=["GET", "POST"])
def choose_course():
    try:
        sid = session["id"]
        dict = {}
        if request.method == "POST":
            cid = request.form.get("cid")
            sc = StudentCourse(s_id=sid, c_id=cid)
            db.session.add(sc)
            db.session.commit()

        now_have_courses_sc = StudentCourse.query.filter(StudentCourse.s_id == sid).all()
        cids = []
        for sc in now_have_courses_sc:
            cids.append(sc.c_id)
        not_hava_courses = Course.query.filter(Course.c_id.notin_(cids), Course.flag == "可选课").all()
        for ncourse in not_hava_courses:
            teacher = Teacher.query.filter(Teacher.t_id == ncourse.t_id).first()
            dict[ncourse] = teacher
        return render_template("student/choose_course.html", dict=dict)
    except Exception:
        flash("出发错误操作")
        return redirect(url_for("student.home"))


@student.route("/unchoose_course", methods=["GET", "POST"])
def unchoose_course():
    try:
        sid = session["id"]
        dict = {}
        if request.method == "POST":
            cid = request.form.get("cid")
            sc = StudentCourse.query.filter(StudentCourse.c_id == cid, StudentCourse.s_id == sid).first()
            db.session.delete(sc)
            db.session.commit()
        now_have_courses_sc = StudentCourse.query.filter(StudentCourse.s_id == sid).all()
        cids = []
        for sc in now_have_courses_sc:
            cids.append(sc.c_id)
        hava_courses = Course.query.filter(Course.c_id.in_(cids), Course.flag == "可选课").all()
        for course in hava_courses:
            teacher = Teacher.query.filter(Teacher.t_id == course.t_id).first()
            dict[course] = teacher
        return render_template("student/unchoose_course.html", dict=dict)
    except Exception:
        flash("出发错误操作")
        return redirect(url_for("student.home"))


@student.route("/update_password", methods=["GET", "POST"])
def update_password():
    sid = session["id"]
    student = Student.query.filter(Student.s_id == sid).first()
    if request.method == "POST":
        old = request.form.get("old")
        if old == student.s_password:
            new = request.form.get("new")
            student.s_password = new
            db.session.commit()
            flash("修改成功! ")
        else:
            flash("旧密码错误, 请重试")
    return render_template("student/update_password.html", student=student)
