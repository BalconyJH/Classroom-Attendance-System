import os
import glob
import time
from io import BytesIO
from typing import Union
from urllib.parse import quote

import cv2
import dlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from PIL import Image, ImageDraw, ImageFont
from flask import Response, Blueprint, flash, jsonify, request, session, url_for, redirect, send_file, render_template

from app import db, app, config

from .models import Faces, Course, Student, Teacher, Time_id, Attendance, StudentCourse

teacher = Blueprint("teacher", __name__, static_folder="static")
# 本次签到的所有人员信息
attend_records = []
# 本次签到的开启时间
the_now_time = ""


@teacher.route("/home")
def home():
    flag = session["id"][0]
    courses = {}
    course = db.session.query(Course).filter(Course.t_id == session["id"]).all()
    for c in course:
        num = db.session.query(StudentCourse).filter(StudentCourse.c_id == c.c_id).count()
        courses[c] = num
    return render_template(
        "teacher/teacher_home.html",
        before=session["time"],
        flag=flag,
        name=session["name"],
        courses=courses,
    )


class VideoCamera:
    def __init__(self, logger, video_stream: Union[int, str] = 0):
        self.match_threshold = 0.5
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )
        self.logger = logger
        self.font = cv2.FONT_ITALIC
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(video_stream)

        # 统计 FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # 统计帧数
        self.frame_cnt = 0

        # 用来存储所有录入人脸特征的数组
        self.features_known_list = []
        # 用来存储录入人脸名字
        self.name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # 用来存储当前帧检测出目标的名字
        self.current_frame_name_list = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_feature_list = []

        # 控制再识别的后续帧数
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    def __del__(self):
        self.video.release()

    def get_face_database(self, cid):
        # print(cid)
        # course_sid = SC.query.filter(SC.c_id==cid).all()
        # all_sid = []
        # for sc in course_sid:
        #     all_sid.append(sc.s_id)
        # from_db_all_features = Faces.query.filter(Faces.s_id.in_(all_sid)).all()
        from_db_all_features = Faces.query.all()
        if from_db_all_features:
            for from_db_one_features in from_db_all_features:
                someone_feature_str = str(from_db_one_features.feature).split(",")
                self.name_known_list.append(from_db_one_features.s_id)
                features_someone_arr = []
                for one_feature in someone_feature_str:
                    if one_feature == "":
                        features_someone_arr.append("0")
                    else:
                        features_someone_arr.append(float(one_feature))
                self.features_known_list.append(features_someone_arr)
            return 1
        else:
            return 0

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd: np.ndarray):
        cv2.putText(
            img_rd,
            "FPS:   " + str(self.fps.__round__(2)),
            (20, 100),
            self.font,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    def draw_name(self, img_rd: np.ndarray) -> np.ndarray:
        if self.current_frame_face_position_list and self.current_frame_name_list:
            font = ImageFont.truetype("simsun.ttc", 30)
            img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            draw.text(
                xy=self.current_frame_face_position_list[0],
                text=self.current_frame_name_list[0],
                font=font,
            )
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    def get_frame(self, cid):
        # 检查是否能够获取人脸数据库
        if not self.get_face_database(cid):
            return

        stream = self.video
        while stream.isOpened():
            self.frame_cnt += 1
            flag, img_rd = stream.read()
            # 调整图像大小以加快处理速度
            img_rd = cv2.resize(
                img_rd, (int(img_rd.shape[1] * 0.5), int(img_rd.shape[0] * 0.5)), interpolation=cv2.INTER_AREA
            )

            # 检测当前帧的人脸
            faces = self.face_detector(img_rd, 0)
            self.update_face_counts(len(faces))

            filename = "attendance.txt"
            with open(filename, "a", encoding="utf-8") as file:
                if self.faces_count_changed():
                    # 人脸数量发生变化时的处理
                    self.handle_faces_count_change(faces, img_rd, file)
                else:
                    # 人脸数量未变时的处理
                    self.handle_faces_count_unchanged(faces, img_rd, file)

            img_rd = self.draw_name(img_rd)

            # 添加说明文字和更新FPS
            self.draw_note(img_rd)
            self.update_fps()

            # 返回处理后的图像
            ret, jpeg = cv2.imencode(".jpg", img_rd)
            return jpeg.tobytes()

    def update_face_counts(self, current_count):
        """更新当前帧和上一帧的人脸数量。"""
        self.last_frame_faces_cnt = self.current_frame_face_cnt
        self.current_frame_face_cnt = current_count

    def reset_face_lists(self):
        """重置与人脸识别相关的列表。"""
        self.current_frame_face_position_list = []
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_feature_list = []
        self.current_frame_name_list = []

    def recognize_faces(self, faces, img_rd, file):
        """
        识别并处理当前帧中的人脸。

        参数:
            faces: 当前帧检测到的人脸列表。
            img_rd: 当前帧图像，用于特征提取。
            file: 出勤记录文件，用于记录识别的人脸。

        处理流程:
            1. 对每个检测到的人脸，提取特征。
            2. 将提取的特征与已知特征库进行匹配。
            3. 处理匹配结果，更新人脸识别信息。
        """
        for face in faces:
            # 提取人脸特征
            shape = self.shape_predictor(img_rd, face)
            face_descriptor = self.face_recognition_model.compute_face_descriptor(img_rd, shape)

            # 将当前人脸特征转换为numpy数组
            face_feature = np.array(face_descriptor)

            # 初始化匹配结果
            match_name = "Unknown"
            min_distance = float("inf")

            # 特征匹配
            for i, known_feature in enumerate(self.features_known_list):
                distance = np.linalg.norm(face_feature - known_feature)
                if distance < min_distance and distance < self.match_threshold:  # self.match_threshold是匹配阈值
                    min_distance = distance
                    match_name = self.name_known_list[i]  # 假设name_known_list存储了与features_known_list对应的名字

            # 处理匹配结果
            self.logger.info(f"Face matched with {match_name}, distance: {min_distance}")
            if match_name != "Unknown":
                pass
                # with open(file, "a") as f:
                #     f.write(f"{match_name}\n")

    def faces_count_changed(self):
        """检查当前帧和上一帧的人脸数量是否发生变化。"""
        return self.current_frame_face_cnt != self.last_frame_faces_cnt

    def handle_faces_count_change(self, faces, img_rd, file):
        """处理人脸数量变化的情况。"""
        self.reset_face_lists()
        if self.current_frame_face_cnt > 0:
            self.recognize_faces(faces, img_rd, file)
        else:
            self.reclassify_interval_cnt = 0

    def handle_faces_count_unchanged(self, faces, img_rd, file):
        """处理人脸数量未变的情况。"""
        if "unknown" in self.current_frame_name_list:
            self.reclassify_interval_cnt += 1
        if self.current_frame_face_cnt == 1 and self.reclassify_interval_cnt == self.reclassify_interval:
            self.recognize_faces(faces, img_rd, file)


# 老师端
@teacher.route("/reco_faces")
def reco_faces():
    return render_template("teacher/index.html")


def stream_video_frames(camera: VideoCamera, course_id: str):
    """
    生成视频流的帧数据
    :param camera: VideoCamera对象
    :param course_id: 课程id
    :return: 生成器，每次返回一帧的数据
    """
    while True:
        with app.app_context():
            frame = camera.get_frame(course_id)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@teacher.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    return Response(
        stream_video_frames(VideoCamera(logger=app.logger), session["course"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@teacher.route("/all_course")
def all_course():
    teacher_all_course = Course.query.filter(Course.t_id == session["id"])
    return render_template("teacher/course_attend.html", courses=teacher_all_course)


# 开启签到
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
    """初始化考勤记录，标记所有学生为未签到"""
    the_course_students = StudentCourse.query.filter_by(c_id=cid)
    all_students_attend = [Attendance(s_id=sc.s_id, c_id=cid, time=now, result="缺勤") for sc in the_course_students]
    db.session.add_all(all_students_attend)
    db.session.commit()


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

    return render_template("teacher/index.html")


# 实时显示当前签到人员
@teacher.route("/now_attend")
def now_attend():
    return jsonify(attend_records)


# 停止签到
@teacher.route("/stop_records", methods=["POST"])
def stop_records():
    all_sid = []
    all_cid = session["course"]
    all_time = session["now_time"]
    for someone_attend in attend_records:
        sid = someone_attend.split("  ")[0]

        all_sid.append(sid)
    Attendance.query.filter(
        Attendance.time == all_time,
        Attendance.c_id == all_cid,
        Attendance.s_id.in_(all_sid),
    ).update({"result": "已签到"}, synchronize_session=False)
    db.session.commit()
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
                        tt = Time_id(id=num, time=times[i])
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
                    tt = Time_id(id=num, time=times[i])
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
                        tt = Time_id(id=num, time=times[i])
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
                    tt = Time_id(id=num, time=times[i])
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
            tt = Time_id(id=num, time=times[i])
            num += 1
            one_course_all_time_attends[tt] = one_time_attends
        dict[course] = one_course_all_time_attends
    return render_template("teacher/show_records.html", dict=dict, courses=courses)


@teacher.route("/update_attend", methods=["POST"])
def update_attend():
    course = request.form.get("course_id")
    time = request.form.get("time")
    sid = request.form.get("sid")
    result = request.form.get("result")
    one_attend = Attendance.query.filter(
        Attendance.c_id == course, Attendance.s_id == sid, Attendance.time == time
    ).first()
    one_attend.result = result
    db.session.commit()
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
def update_password():
    tid = session["id"]
    teacher = Teacher.query.filter(Teacher.t_id == tid).first()
    if request.method == "POST":
        old = request.form.get("old")
        if old == teacher.t_password:
            new = request.form.get("new")
            teacher.t_password = new
            db.session.commit()
            flash("修改成功！")
        else:
            flash("旧密码错误，请重试")
    return render_template("teacher/update_password.html", teacher=teacher)


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
                    Student.s_id == StudentCourse.s_id, StudentCourse.c_id == course.c_id, StudentCourse.s_id == sid
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
                        Student.s_id == StudentCourse.s_id, StudentCourse.c_id == course.c_id, StudentCourse.s_id == sid
                    )
                    .all()
                )
                dict[course] = course_student
        else:
            for course in teacher_all_course:
                course_student = (
                    db.session.query(Student)
                    .join(StudentCourse)
                    .filter(Student.s_id == StudentCourse.s_id, StudentCourse.c_id == course.c_id)
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


def mark_student_as_deleted(sid):
    """标记学生为已删除状态，并提交数据库更改。"""
    student = Student.query.filter(Student.s_id == sid).first()
    if student:
        student.flag = 1  # 假设flag=1表示学生已被删除
        db.session.commit()


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
    ALLOWED_EXTENSIONS = {"xlsx", "xls"}
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


# 检测学号存在
def sid_if_exist(sid):
    num = Student.query.filter(Student.s_id.in_(sid)).count()
    return num


def cid_if_exist(cid):
    num = Course.query.filter(Course.c_id.in_(cid)).count()
    return num


def tid_if_exist(tid):
    num = Teacher.query.filter(Teacher.t_id.in_(tid)).count()
    return num


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
        except Exception as e:
            print("Error:", e)
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
        except Exception as e:
            print("Error:", e)
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
    # url = str(request.url)
    # paras = url.split('?')[1]
    # print(paras)
    # nums = paras.split('&')
    # print(nums)
    # cid = nums[0].split('=')[1]
    # cname = nums[1].split('=')[1]
    # time = nums[2].split('=')[1]
    cid = request.form.get("cid")
    cname = request.form.get("cname")
    time = request.form.get("time")
    # 建立数据库引擎
    engine = create_engine("mysql+pymysql://root:990722@localhost:3306/test?charset=utf8")
    # 写一条sql
    sql = "select s_id 学号,result 考勤结果 from attendance where c_id='" + str(cid) + "' and time='" + str(time) + "'"
    print(sql)
    # 建立dataframe
    df = pd.read_sql_query(sql, engine)
    out = BytesIO()
    writer = pd.ExcelWriter("out.xlsx", engine="xlsxwriter")
    df.to_excel(excel_writer=writer, sheet_name="Sheet1", index=False)
    writer.save()
    out.seek(0)
    # 文件名中文支持
    name = cname + time + "考勤.xlsx"
    file_name = quote(name)
    response = send_file(out, as_attachment=True, attachment_filename=file_name)
    response.headers["Content-Disposition"] += f"; filename*=utf-8''{file_name}"
    return send_file("../out.xlsx", as_attachment=True, attachment_filename=file_name)
