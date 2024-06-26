import os
import glob
import time
from io import BytesIO
from typing import Any
from urllib.parse import quote

import pandas as pd
from loguru import logger
from flask import Response, Blueprint, flash, jsonify, request, session, url_for, redirect, send_file, render_template

from app.utils.camera import VideoCamera
from app import TeacherSession, db, app, config
from app.utils.session_manager import SessionManager
from app.data_access.student_repository import update_user_password
from app.database.models import Course, TimeID, Student, Teacher, Attendance, StudentCourse
from app.data_access.teacher_repository import (
    cid_if_exist,
    sid_if_exist,
    tid_if_exist,
    update_attendance,
    update_course_times,
    mark_student_as_deleted,
    get_courses_by_teacher_id,
    update_attendance_records,
    initialize_attendance_records,
    get_student_count_by_course_id,
)

teacher = Blueprint("teacher", __name__, static_folder="static")
# 本次签到的所有人员信息
attend_records = []
# 本次签到的开启时间
the_now_time = ""


@teacher.route("/home")
async def home() -> Any:
    """渲染教师首页模板, 包括课程信息和其他相关数据。

    返回:
        渲染的教师首页模板。
    """
    teacher_session: TeacherSession = SessionManager.load_session_data()
    courses = {}
    teacher_id = teacher_session.id
    flag = teacher_id[0]
    teacher_courses = await get_courses_by_teacher_id(teacher_id)
    for course in teacher_courses:
        num = await get_student_count_by_course_id(course.c_id)
        courses[course] = num
    return render_template(
        "teacher/teacher_home.html",
        before=session["time"],
        flag=flag,
        name=session["name"],
        courses=courses,
    )


# 老师端
@teacher.route("/reco_faces")
def reco_faces():
    return render_template("teacher/index.html")


def stream_video_frames(camera: VideoCamera, course_id: str, timeout_seconds=10):
    """
    生成视频流的帧数据, 带有超时机制。
    :param camera: VideoCamera对象
    :param course_id: 课程id
    :param timeout_seconds: 超时时间(秒)
    :return: 生成器, 每次返回一帧的数据
    """
    with app.app_context():
        while True:
            frame, student_id = camera.get_frame()
            if student_id != "Unknown":
                logger.debug(f"检测到学生: {student_id}")
                update_attend_records(student_id)
            if frame is not None:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@teacher.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    return Response(
        stream_video_frames(VideoCamera(), session["course"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@teacher.route("/all_course")
def all_course():
    teacher_all_course = Course.query.filter(Course.t_id == session["id"])
    return render_template("teacher/course_attend.html", courses=teacher_all_course)


@teacher.route("/records", methods=["POST"])
def records():
    cid = request.form.get("id")
    if not cid:
        flash("课程号不能为空")

    session["course"] = cid
    now = update_course_times(cid)
    if not now:
        flash("课程号不存在")

    session["now_time"] = now
    initialize_attendance_records(cid, now)
    signed_in_students = attend_records

    return render_template("teacher/index.html", signed_in_students=signed_in_students)


# 实时显示当前签到人员
@teacher.route("/now_attend")
def now_attend():
    return jsonify(attend_records)


# 停止签到
@teacher.route("/stop_records", methods=["POST"])
async def stop_records():
    all_sid = []
    cid = session["course"]
    all_time = session["now_time"]
    for someone_attend in attend_records:
        sid = someone_attend.split("  ")[0]
        all_sid.append(sid)

    logger.debug(f"本次签到的所有学生ID: {all_sid}")
    logger.debug(f"本次签到的课程ID: {cid}")
    await update_attendance_records(all_sid, cid, all_time)
    return redirect(url_for("teacher.all_course"))


@teacher.route("/select_all_records", methods=["GET", "POST"])
def select_all_records():
    tid = session["id"]
    dict = {}
    num = 0
    if request.method == "POST":
        cid = request.form.get("course_id")
        sid = request.form.get("sid")
        select_time = request.form.get("time")
        if cid != "" and select_time != "":
            courses = db.session.query(Course).filter(Course.t_id == tid, Course.c_id == cid)
            num = 0
            for course in courses:
                times = course.times.split("/")
                one_course_all_time_attends = {}
                for i in range(1, len(times)):
                    one_time = times[i].split(" ")[0]
                    if one_time == select_time:
                        if sid != "":
                            one_time_attends = (
                                db.session.query(Attendance)
                                .filter(
                                    Attendance.c_id == course.c_id,
                                    Attendance.time == times[i],
                                    Attendance.s_id == sid,
                                )
                                .order_by("s_id")
                                .all()
                            )
                        else:
                            one_time_attends = (
                                db.session.query(Attendance)
                                .filter(
                                    Attendance.c_id == course.c_id,
                                    Attendance.time == times[i],
                                )
                                .order_by("s_id")
                                .all()
                            )
                        tt = TimeID(id=num, time=times[i])
                        num += 1
                        one_course_all_time_attends[tt] = one_time_attends
                dict[course] = one_course_all_time_attends
            courses = db.session.query(Course).filter(Course.t_id == tid)
            return render_template("teacher/show_records.html", dict=dict, courses=courses)
        elif cid != "" and select_time == "":
            courses = db.session.query(Course).filter(Course.t_id == tid, Course.c_id == cid)
            num = 0
            for course in courses:
                times = course.times.split("/")
                one_course_all_time_attends = {}
                for i in range(1, len(times)):
                    if sid == "":
                        one_time_attends = (
                            db.session.query(Attendance)
                            .filter(
                                Attendance.c_id == course.c_id,
                                Attendance.time == times[i],
                            )
                            .order_by("s_id")
                            .all()
                        )
                    else:
                        one_time_attends = (
                            db.session.query(Attendance)
                            .filter(
                                Attendance.c_id == course.c_id,
                                Attendance.time == times[i],
                                Attendance.s_id == sid,
                            )
                            .order_by("s_id")
                            .all()
                        )
                    tt = TimeID(id=num, time=times[i])
                    num += 1
                    one_course_all_time_attends[tt] = one_time_attends
                dict[course] = one_course_all_time_attends
            return render_template("teacher/show_records.html", dict=dict, courses=courses)
        elif cid == "" and select_time != "":
            courses = db.session.query(Course).filter(Course.t_id == tid)
            num = 0
            for course in courses:
                times = course.times.split("/")
                one_course_all_time_attends = {}
                for i in range(1, len(times)):
                    one_time = times[i].split(" ")[0]
                    if one_time == select_time:
                        if sid != "":
                            one_time_attends = (
                                db.session.query(Attendance)
                                .filter(
                                    Attendance.c_id == course.c_id,
                                    Attendance.time == times[i],
                                    Attendance.s_id == sid,
                                )
                                .order_by("s_id")
                                .all()
                            )
                        else:
                            one_time_attends = (
                                db.session.query(Attendance)
                                .filter(
                                    Attendance.c_id == course.c_id,
                                    Attendance.time == times[i],
                                )
                                .order_by("s_id")
                                .all()
                            )
                        tt = TimeID(id=num, time=times[i])
                        num += 1
                        one_course_all_time_attends[tt] = one_time_attends
                dict[course] = one_course_all_time_attends
            return render_template("teacher/show_records.html", dict=dict, courses=courses)
        else:  # cid = '' select=''
            courses = db.session.query(Course).filter(Course.t_id == tid)
            num = 0
            for course in courses:
                times = course.times.split("/")
                one_course_all_time_attends = {}
                for i in range(1, len(times)):
                    if sid == "":
                        one_time_attends = (
                            db.session.query(Attendance)
                            .filter(
                                Attendance.c_id == course.c_id,
                                Attendance.time == times[i],
                            )
                            .order_by("s_id")
                            .all()
                        )
                    else:
                        one_time_attends = (
                            db.session.query(Attendance)
                            .filter(
                                Attendance.c_id == course.c_id,
                                Attendance.time == times[i],
                                Attendance.s_id == sid,
                            )
                            .all()
                        )
                    tt = TimeID(id=num, time=times[i])
                    num += 1
                    one_course_all_time_attends[tt] = one_time_attends
                dict[course] = one_course_all_time_attends
            return render_template("teacher/show_records.html", dict=dict, courses=courses)
    courses = db.session.query(Course).filter(Course.t_id == tid)
    num = 0
    for course in courses:
        times = course.times.split("/")
        one_course_all_time_attends = {}
        for i in range(1, len(times)):
            one_time_attends = (
                db.session.query(Attendance)
                .filter(Attendance.c_id == course.c_id, Attendance.time == times[i])
                .order_by("s_id")
                .all()
            )
            tt = TimeID(id=num, time=times[i])
            num += 1
            one_course_all_time_attends[tt] = one_time_attends
        dict[course] = one_course_all_time_attends
    return render_template("teacher/show_records.html", dict=dict, courses=courses)


@teacher.route("/update_attend", methods=["POST"])
async def update_attend():
    course_id = request.form.get("course_id")
    now_time = request.form.get("time")
    student_id = request.form.get("sid")
    result = request.form.get("result")
    await update_attendance(course_id=course_id, student_id=student_id, class_time=now_time, result=result)
    return redirect(url_for("teacher.select_all_records"))


@teacher.route("/course_management", methods=["GET", "POST"])
def course_management():
    dict = {}
    if request.method == "POST":
        cid = request.form.get("course_id")
        request.form.get("course_name")
        sid = request.form.get("sid")
        sc = StudentCourse.query.filter(StudentCourse.c_id == cid, StudentCourse.s_id == sid).first()
        db.session.delete(sc)
        db.session.commit()
    teacher_all_course = Course.query.filter(Course.t_id == session["id"])
    for course in teacher_all_course:
        course_student = (
            db.session.query(Student)
            .join(StudentCourse)
            .filter(Student.s_id == StudentCourse.s_id, StudentCourse.c_id == course.c_id)
            .all()
        )
        dict[course] = course_student
    return render_template("teacher/course_management.html", dict=dict)


@teacher.route("/new_course", methods=["POST"])
def new_course():
    max = db.session.query(Course).order_by(Course.c_id.desc()).first()
    if max:
        cid = int(max.c_id) + 1
        cid = str(cid)
    else:
        cid = str(100001)
    course = Course(c_id=cid, c_name=request.form.get("cname"), t_id=session["id"])
    db.session.add(course)
    db.session.commit()
    return redirect(url_for("teacher.course_management"))


@teacher.route("/open_course", methods=["POST"])
def open_course():
    cid = request.form.get("course_id")
    course = Course.query.filter(Course.c_id == cid).first()
    course.flag = "可选课"
    db.session.commit()
    return redirect(url_for("teacher.course_management"))


@teacher.route("/close_course", methods=["POST"])
def close_course():
    cid = request.form.get("course_id")
    course = Course.query.filter(Course.c_id == cid).first()
    course.flag = "不可选课"
    db.session.commit()
    return redirect(url_for("teacher.course_management"))


@teacher.route("/update_password", methods=["GET", "POST"])
async def update_password():
    user_id = session["id"]
    if request.method == "POST":
        old = request.form.get("old")
        new = request.form.get("new")
        await update_user_password("teacher", user_id, old, new)
    return render_template(
        "teacher/update_password.html",
        teacher=Teacher.query.filter(Teacher.t_id == user_id).first(),
    )


@teacher.route("/select_sc", methods=["GET", "POST"])
def select_sc():
    dict = {}
    teacher_all_course = Course.query.filter(Course.t_id == session["id"])
    if request.method == "POST":
        cid = request.form.get("course_id")
        sid = request.form.get("sid")
        if cid != "" and sid != "":
            course = Course.query.filter(Course.c_id == cid).first()
            dict[course] = (
                db.session.query(Student)
                .join(StudentCourse)
                .filter(
                    Student.s_id == StudentCourse.s_id,
                    StudentCourse.c_id == course.c_id,
                    StudentCourse.s_id == sid,
                )
                .all()
            )
        elif cid != "" and sid == "":
            course = Course.query.filter(Course.c_id == cid).first()
            dict[course] = (
                db.session.query(Student)
                .join(StudentCourse)
                .filter(Student.s_id == StudentCourse.s_id, StudentCourse.c_id == cid)
                .all()
            )
        elif cid == "" and sid != "":
            for course in teacher_all_course:
                course_student = (
                    db.session.query(Student)
                    .join(StudentCourse)
                    .filter(
                        Student.s_id == StudentCourse.s_id,
                        StudentCourse.c_id == course.c_id,
                        StudentCourse.s_id == sid,
                    )
                    .all()
                )
                dict[course] = course_student
        else:
            for course in teacher_all_course:
                course_student = (
                    db.session.query(Student)
                    .join(StudentCourse)
                    .filter(
                        Student.s_id == StudentCourse.s_id,
                        StudentCourse.c_id == course.c_id,
                    )
                    .all()
                )
                dict[course] = course_student
        return render_template("teacher/student_getFace.html", dict=dict, courses=teacher_all_course)
    for course in teacher_all_course:
        course_student = (
            db.session.query(Student)
            .join(StudentCourse)
            .filter(Student.s_id == StudentCourse.s_id, StudentCourse.c_id == course.c_id)
            .all()
        )
        dict[course] = course_student
    return render_template("teacher/student_getFace.html", dict=dict, courses=teacher_all_course)


@teacher.route("/open_getFace", methods=["POST"])
def open_getFace():
    sid = request.form.get("sid")
    student = Student.query.filter(Student.s_id == sid).first()
    student.flag = 1
    db.session.commit()
    return redirect(url_for("teacher.select_sc"))


@teacher.route("/close_getFace", methods=["POST"])
def close_getFace():
    sid = request.form.get("sid")
    student = Student.query.filter(Student.s_id == sid).first()
    student.flag = 0
    db.session.commit()
    return redirect(url_for("teacher.select_sc"))


def delete_all_images_in_folder(uid):
    folder_path = os.path.join(config.cache_path, "dataset", uid)
    if os.path.exists(folder_path):
        images = glob.glob(os.path.join(str(folder_path), "*.jpg"))
        for image in images:
            os.remove(image)
        os.rmdir(folder_path)


@teacher.route("/delete_face", methods=["POST"])
def delete_face():
    sid = request.form.get("sid")
    # 拆分的数据库逻辑
    mark_student_as_deleted(sid)
    # 删除文件的逻辑
    delete_all_images_in_folder(sid)
    return redirect(url_for("teacher.select_sc"))


def allowed_file(filename):
    allowed_extensions = {"xlsx", "xls"}
    return "." in filename and filename.rsplit(".", 1)[1] in allowed_extensions


@teacher.route("upload_sc", methods=["POST"])
def upload_sc():
    sc_file = request.files.get("sc_file")
    if sc_file:
        if allowed_file(sc_file.filename):
            sc_file.save(sc_file.filename)
            df = pd.DataFrame(pd.read_excel(sc_file.filename))
            df1 = df[["学号", "课程号"]]
            sid = df1[["学号"]].values.T.tolist()[:][0]
            cid = df1[["课程号"]].values.T.tolist()[:][0]
            if df.isnull().values.any():
                flash("存在空信息")
            else:
                sid_diff = len(set(sid)) - sid_if_exist(sid)
                cid_diff = len(set(cid)) - cid_if_exist(cid)
                if sid_diff == 0 and cid_diff == 0:
                    flash("success")
                    for i in range(len(sid)):
                        sc = StudentCourse(s_id=sid[i], c_id=cid[i])
                        db.session.merge(sc)
                        i += 1
                    db.session.commit()

                elif sid_diff == 0 and cid_diff != 0:
                    flash("有课程号不存在")
                elif sid_diff != 0 and cid_diff == 0:
                    flash("有学号不存在")
                else:
                    flash("有学号、课程号不存在")
            os.remove(sc_file.filename)
        else:
            flash("只能识别xlsx,xls文件")
    else:
        flash("请选择文件")
    return redirect(url_for("teacher.course_management"))


@teacher.route("/select_all_teacher", methods=["POST", "GET"])
def select_all_teacher():
    if request.method == "POST":
        try:
            id = request.form.get("id")
            flag = request.form.get("flag")
            teacher = Teacher.query.get(id)
            if flag:
                sc = (
                    db.session.query(StudentCourse)
                    .join(Course)
                    .filter(StudentCourse.c_id == Course.c_id, Course.t_id == id)
                    .all()
                )
                [db.session.delete(u) for u in sc]
                Course.query.filter(Course.t_id == id).delete()
            db.session.delete(teacher)
            db.session.commit()
        except Exception:
            flash("出发错误操作")
            return redirect(url_for("teacher.home"))
    teachers = Teacher.query.all()
    dict = {}
    for t in teachers:
        tc = Course.query.filter(Course.t_id == t.t_id).all()
        if tc:
            dict[t] = 1
        else:
            dict[t] = 0
    return render_template("teacher/all_teacher.html", dict=dict)


@teacher.route("/select_all_student", methods=["POST", "GET"])
def select_all_student():
    if request.method == "POST":
        try:
            id = request.form.get("id")
            flag = request.form.get("flag")
            student = Student.query.get(id)
            if flag:
                StudentCourse.query.filter(StudentCourse.s_id == id).delete()
            db.session.delete(student)
            db.session.commit()
        except Exception:
            flash("出发错误操作")
            return redirect(url_for("teacher.home"))
    students = Student.query.all()
    dict = {}
    for s in students:
        tc = StudentCourse.query.filter(StudentCourse.s_id == s.s_id).all()
        if tc:
            dict[s] = 1
        else:
            dict[s] = 0
    return render_template("teacher/all_student.html", dict=dict)


@teacher.route("/upload_teacher", methods=["POST"])
def upload_teacher():
    file = request.files.get("teacher_file")
    if file:
        if allowed_file(file.filename):
            file.save(file.filename)
            df = pd.DataFrame(pd.read_excel(file.filename))
            df1 = df[["工号", "姓名", "密码"]]
            id = df1[["工号"]].values.T.tolist()[:][0]
            name = df1[["姓名"]].values.T.tolist()[:][0]
            pwd = df1[["密码"]].values.T.tolist()[:][0]
            if df.isnull().values.any() or len(id) == 0:
                flash("存在空信息")
            else:
                tid_diff = tid_if_exist(id)
                if tid_diff != 0:
                    flash("工号存在重复")
                else:
                    flash("success")
                    for i in range(len(id)):
                        t = Teacher(t_id=id[i], t_name=name[i], t_password=pwd[i])
                        db.session.add(t)
                        i += 1
                    db.session.commit()
            os.remove(file.filename)

        else:
            flash("只能识别'xlsx,xls'文件")
    else:
        flash("请选择文件")
    return redirect(url_for("teacher.select_all_teacher"))


@teacher.route("/upload_student", methods=["POST"])
def upload_student():
    file = request.files.get("student_file")
    if file:
        if allowed_file(file.filename):
            file.save(file.filename)
            df = pd.DataFrame(pd.read_excel(file.filename))
            df1 = df[["学号", "姓名", "密码"]]
            id = df1[["学号"]].values.T.tolist()[:][0]
            name = df1[["姓名"]].values.T.tolist()[:][0]
            pwd = df1[["密码"]].values.T.tolist()[:][0]
            if df.isnull().values.any() or len(id) == 0:
                flash("存在空信息")
            else:
                sid_diff = sid_if_exist(id)
                if sid_diff != 0:
                    flash("学号存在重复")
                else:
                    flash("success")
                    for i in range(len(id)):
                        s = Student(s_id=id[i], s_name=name[i], s_password=pwd[i])
                        db.session.add(s)
                        i += 1
                    db.session.commit()
            os.remove(file.filename)

        else:
            flash("只能识别'xlsx,xls'文件")
    else:
        flash("请选择文件")

    return redirect(url_for("teacher.select_all_student"))


@teacher.route("/download", methods=["POST"])
def download():
    cid = request.form.get("cid")
    cname = request.form.get("cname")
    attendance_time = request.form.get("time")

    query = Attendance.query.filter_by(c_id=cid, time=attendance_time).all()
    df = pd.DataFrame([(d.s_id, d.result) for d in query], columns=["学号", "考勤结果"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    output.seek(0)

    file_name = quote(f"{cname}{attendance_time}考勤.xlsx")

    return send_file(
        output,
        download_name=file_name,
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def update_attend_records(sid: str) -> None:
    """更新考勤记录"""
    sid_exists = False
    for record in attend_records:
        if str(sid) == record.split()[0]:
            sid_exists = True
            break

    if not sid_exists:
        attend_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        attend_records.append(f"{sid}  {attend_time}  已签到\n")
