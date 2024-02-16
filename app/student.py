import base64
from datetime import datetime
from pathlib import Path

from flask import Blueprint, flash, request, session, url_for, redirect, render_template

from app import app
from app.config import config
from app.face_feature_processor import FaceFeatureProcessor
from app.database.models import Student
from app.data_access.student_repository import (
    get_student_by_id,
    get_attendance_records,
    get_monthly_attendance_summary,
    pre_work_mkdir,
    extract_and_process_features,
    update_database_with_features,
    get_courses_by_student_id,
    get_records_by_course_and_time,
    add_student_course,
    get_available_courses,
    delete_student_course,
    get_selectable_courses_and_teachers,
    update_user_password,
)

student = Blueprint("student", __name__, static_folder="static")


async def save_image(data: bytes, path: Path) -> None:
    """
    保存图片
    :param data: 图片数据
    :param path: 保存路径
    :return: None
    """
    with open(path, "wb") as file:
        file.write(data)


@student.route("/home")
async def home():
    student_id = session["id"]
    student_instance = await get_student_by_id(student_id)
    session["flag"] = student_instance.flag
    records = await get_attendance_records(student_id)
    month = datetime.now().month
    year = datetime.now().year
    num = await get_monthly_attendance_summary(student_id, month, year)

    return render_template(
        "student/student_home.html",
        flag=session["flag"],
        before=session["time"],
        records=records,
        name=session["name"],
        num=num,
    )


@student.route("/get_faces", methods=["GET", "POST"])
async def get_faces():
    if request.method == "POST":
        imgdata = base64.b64decode(request.form.get("face"))
        path = Path(config.cache_path / "dataset" / session["id"])
        face_register = FaceFeatureProcessor(app.logger)
        if session["num"] == 0:
            await pre_work_mkdir(path)
        if session["num"] == 5:
            session["num"] = 0
        session["num"] += 1
        current_face_path = path / f"{session['num']}.jpg"
        await save_image(imgdata, current_face_path)
        flag = face_register.process_single_face_image(str(current_face_path))
        if flag != "right":
            session["num"] -= 1
        return {"result": flag, "code": session["num"]}
    return render_template("student/get_faces.html")


@student.route("/upload_faces", methods=["POST"])
async def upload_faces():
    try:
        # 提取特征并处理数据
        features = await extract_and_process_features()
        # 更新数据库
        await update_database_with_features(features)
        # 设置成功消息并重定向
        flash("提交成功! ")
        return redirect(url_for("student.home"))
    except Exception as e:
        app.logger.debug("Error:", e)
        flash("提交不合格照片, 请拍摄合格后再重试")
        return redirect(url_for("student.home"))


@student.route("/my_faces")
async def my_faces():
    current_face_path = Path(config.cache_path) / "dataset" / session["id"]

    face_path = Path("static/caches/dataset") / session["id"]

    photos_list = list(current_face_path.glob("*.jpg"))
    num = len(photos_list)

    paths = [str(face_path / f"{i + 1}.jpg") for i in range(num)]

    return render_template("student/my_faces.html", face_paths=paths)


@student.route("/my_records", methods=["GET", "POST"])
async def my_records():
    sid = session["id"]
    courses = await get_courses_by_student_id(sid)
    if request.method == "POST":
        cid = str(request.form.get("course_id", ""))
        time = str(request.form.get("time", ""))
        records_dict = await get_records_by_course_and_time(sid, cid if cid else None, time if time else None)
    else:
        records_dict = await get_records_by_course_and_time(sid)
    return render_template("student/my_records.html", dict=records_dict, courses=courses)


@student.route("/choose_course", methods=["GET", "POST"])
async def choose_course():
    try:
        sid = session["id"]
        if request.method == "POST":
            cid = request.form.get("cid")
            if not await add_student_course(sid, cid):
                flash("选课失败, 请重试")
                return redirect(url_for("student.choose_course"))

        available_courses = await get_available_courses(sid)
        return render_template("student/choose_course.html", dict=available_courses)
    except Exception:
        flash("未知错误操作")
        return redirect(url_for("student.home"))


@student.route("/drop_course", methods=["GET", "POST"])
async def drop_course():
    try:
        sid = session["id"]
        if request.method == "POST":
            cid = request.form.get("cid")
            await delete_student_course(sid, cid)  # 删除选课记录

        selectable_courses = await get_selectable_courses_and_teachers(sid)  # 获取可选课程及其教师

        return render_template("student/drop_course.html", dict=selectable_courses)
    except Exception:
        flash("未知错误")
        return redirect(url_for("student.home"))


@student.route("/update_password", methods=["GET", "POST"])
async def update_password():
    user_id = session["id"]
    if request.method == "POST":
        old = request.form.get("old")
        new = request.form.get("new")
        if await update_user_password("student", user_id, old, new):
            app.logger.debug(f"用户:{user_id}修改密码 成功")
        else:
            app.logger.debug(f"用户:{user_id}修改密码 失败")
    return render_template(
        "student/update_password.html", student=Student.query.filter(Student.s_id == user_id).first()
    )


# @app.errorhandler(404)
# async def page_not_found(e):
#     app.logger.error("404 error: ", e)
#     return render_template("404.html"), 404


# @app.errorhandler(Exception)
# async def handle_exception(e):
#     app.logger.error("An error occurred: ", e)
#     return render_template("error.html"), 500
