import os
import sys
import subprocess
from flask import Flask, render_template, request, redirect, url_for, session, flash

from settings import ADMIN_PASSWORD, SECRET_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "logs", "access_log.txt")

app = Flask(__name__)
app.secret_key = SECRET_KEY


def is_admin_logged_in():
    return session.get("admin_logged_in", False)


def run_script_background(script_name, args=None):
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, script_path]
    if args:
        command.extend(args)

    subprocess.Popen(command, cwd=BASE_DIR)


def run_script_in_terminal(script_name, args=None):
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    python_path = sys.executable

    command_parts = [f'"{python_path}"', f'"{script_path}"']
    if args:
        command_parts.extend(f'"{arg}"' for arg in args)

    shell_command = " ".join(command_parts)
    terminal_command = f'cd "{BASE_DIR}" && {shell_command}'
    escaped_command = terminal_command.replace('"', '\\"')

    applescript = f'''
tell application "Terminal"
    activate
    do script "{escaped_command}"
end tell
'''

    subprocess.Popen(["osascript", "-e", applescript])


@app.route("/")
def home():
    if is_admin_logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "").strip()

        if password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            flash("Admin login successful.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Incorrect password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not is_admin_logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if not is_admin_logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":
        person_name = request.form.get("person_name", "").strip()

        if not person_name:
            flash("Please enter a person's name.", "error")
            return redirect(url_for("register"))

        try:
            run_script_in_terminal("enroll.py", [person_name])
            flash(f"Enrollment started for {person_name}. Check the new Terminal window.", "success")
            return redirect(url_for("dashboard"))
        except Exception as error:
            flash(f"Could not start enrollment: {error}", "error")

    return render_template("register.html")


@app.route("/rebuild-database", methods=["POST"])
def rebuild_database():
    if not is_admin_logged_in():
        return redirect(url_for("login"))

    try:
        run_script_background("build_database.py")
        flash("Embedding database rebuild started.", "success")
    except Exception as error:
        flash(f"Could not rebuild database: {error}", "error")

    return redirect(url_for("dashboard"))


@app.route("/start-access-gate", methods=["POST"])
def start_access_gate():
    if not is_admin_logged_in():
        return redirect(url_for("login"))

    try:
        run_script_in_terminal("app.py")
        flash("Access gate started. Check the new Terminal window.", "success")
    except Exception as error:
        flash(f"Could not start access gate: {error}", "error")

    return redirect(url_for("dashboard"))


@app.route("/logs")
def view_logs():
    if not is_admin_logged_in():
        return redirect(url_for("login"))

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            logs = file.read()
    else:
        logs = "No logs found yet."

    return render_template("logs.html", logs=logs)


if __name__ == "__main__":
    app.run(debug=True)