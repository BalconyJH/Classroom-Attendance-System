from datetime import datetime
from logging.config import dictConfig

import dlib
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, flash, request, session, url_for, redirect, render_template

from config import config

config_dict = config.dict()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config_dict.secret_key.get_secret_value(),
    SQLALCHEMY_DATABASE_URI=config_dict.sqlalchemy_database_uri.get_secret_value(),
    SQLALCHEMY_TRACK_MODIFICATIONS=config_dict.sqlalchemy_track_modifications
)
db = SQLAlchemy(app)
from .models import Student, Teacher  # noqa E402

# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
#     datefmt='%d/%b/%Y %H:%M:%S'
# )
# handler.setFormatter(formatter)
# if app.logger.hasHandlers():
#     app.logger.handlers.clear()
# app.logger.addHandler(handler)
print(f"CUDA STATUS:{dlib.DLIB_USE_CUDA}")
dictConfig({
    "version": 1,
    "formatters": {"default": {
        "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    }},
    "handlers": {"wsgi": {
        "class": "logging.StreamHandler",
        "stream": "ext://flask.logging.wsgi_errors_stream",
        "formatter": "default"
    }},
    "root": {
        "level": "INFO",
        "handlers": ["wsgi"]
    }
})


@app.route("/", methods=["GET", "POST"])
def login():
    app.logger.debug("A value for debugging")
    app.logger.info("Info level log")
    app.logger.warning("A warning occurred (%d apples)", 42)
    app.logger.error("An error occurred")
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if len(username) == 13:
            students = Student.query.filter(Student.s_id == username).first()
            if students:
                if students.s_password == password:
                    flash("登陆成功")
                    session["username"] = username
                    session["id"] = students.s_id
                    session["num"] = 0  # students.num
                    session["name"] = students.s_name
                    session["role"] = "student"
                    session["flag"] = students.flag
                    if students.before:
                        session["time"] = students.before
                    else:
                        session["time"] = time
                    students.before = time
                    db.session.commit()
                    return redirect(url_for("student.home"))
                else:
                    flash("密码错误，请重试")
            else:
                flash("学号错误，请重试")
        elif len(username) == 8:
            teachers = Teacher.query.filter(Teacher.t_id == username).first()
            if teachers:
                if teachers.t_password == password:
                    flash("登陆成功")
                    session["username"] = username
                    session["id"] = teachers.t_id
                    session["name"] = teachers.t_name
                    session["role"] = "teacher"
                    session["attend"] = []
                    if teachers.before:
                        session["time"] = teachers.before
                    else:
                        session["time"] = time
                    teachers.before = time
                    db.session.commit()
                    return redirect(url_for("teacher.home"))
                else:
                    flash("密码错误，请重试")
            else:
                flash("工号错误，请重试")
        else:
            flash("账号不合法，请用学号/工号登录")
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
