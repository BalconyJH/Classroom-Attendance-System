from datetime import datetime

import sentry_sdk

from config import config
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, flash, request, session, url_for, redirect, render_template

config_dict = config.dict()


sentry_sdk.init(
    dsn=config_dict["sentry_dsn"].get_secret_value(),
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
    enable_tracing=config_dict["enable_tracing"],
)


app = Flask(__name__)
app.config.update(
    SECRET_KEY=config_dict["secret_key"].get_secret_value(),
    SQLALCHEMY_DATABASE_URI=config_dict["sqlalchemy_database_uri"].get_secret_value(),
    SQLALCHEMY_TRACK_MODIFICATIONS=config_dict["sqlalchemy_track_modifications"],
)
db = SQLAlchemy(app)
from app import views  # noqa E402

from database.models import Student, Teacher  # E402


async def login_user(user_type: str, username: str, password: str, time: str) -> bool:
    user_model = Student if user_type == "student" else Teacher
    user_id_field = "s_id" if user_type == "student" else "t_id"
    password_field = "s_password" if user_type == "student" else "t_password"

    user = user_model.query.filter_by(**{user_id_field: username}).first()
    if user and getattr(user, password_field) == password:
        session_data = {
            "username": username,
            "id": getattr(user, user_id_field),
            "name": user.s_name if user_type == "student" else user.t_name,
            "role": user_type,
            "time": getattr(user, "before", None) or time,
        }
        if user_type == "student":
            session_data.update({"num": 0, "flag": user.flag})  # 假设这些是学生特有的会话信息
        else:
            session_data["attend"] = []  # 假设这是教师特有的会话信息

        session.update(session_data)
        setattr(user, "before", time)
        db.session.commit()
        return True
    return False


@app.route("/", methods=["GET", "POST"])
async def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        user_type = "student" if len(username) == 13 else "teacher" if len(username) == 8 else None
        if user_type and await login_user(user_type, username, password, time):
            flash("登录成功")
            return redirect(url_for(f"{session['role']}.home"))
        else:
            flash("用户名或密码错误, 请重试")

    return render_template("login.html")


@app.route("/logout")
def logout():
    # students = Student.query.filter(Student.s_id == session['id']).first()
    # students.num = session['num']
    # db.session.commit()
    session.clear()
    return render_template("login.html")


# 拦截器
@app.before_request
def before():
    list = ["png", "css", "js", "ico", "xlsx", "xls", "jpg"]
    url_after = request.url.split(".")[-1]
    if url_after in list:
        return None
    url = str(request.endpoint)
    if url == "logout":
        return None
    if url == "login":
        if "username" in session:
            return redirect("logout")
        else:
            return None
    else:
        if "username" in session:
            role = url.split(".")[0]
            if role == session["role"]:
                return None
            else:
                new_endpoint = session["role"] + "." + "home"
                flash("权限不足")
                return redirect(url_for(new_endpoint))
        else:
            flash("未登录")
            return redirect("/")


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.ico"))
