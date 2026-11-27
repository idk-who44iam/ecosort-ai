import os
import secrets
import sqlite3
from functools import wraps

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import classifier

instance_path = os.environ.get("ECOSORT_INSTANCE_PATH")
app = Flask(
    __name__,
    instance_path=os.path.abspath(instance_path) if instance_path else None,
)

secret_key = os.environ.get("SECRET_KEY")
if not secret_key and os.environ.get("RAILWAY_ENVIRONMENT_NAME"):
    raise RuntimeError("Set the SECRET_KEY variable in the Railway service.")
app.secret_key = secret_key or secrets.token_hex(32)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
app.config["UPLOAD_FOLDER"] = os.path.join(app.instance_path, "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "avif"}

DB_PATH = os.path.join(app.instance_path, "ecosort.db")
os.makedirs(app.instance_path, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        db.commit()
    finally:
        db.close()


init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Login required decorator
# --------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to use that feature.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_home(result=None):
    """Render the home page with the app's fixed AI model."""
    return render_template(
        "index.html",
        result=result,
        model=classifier.default_model(),
    )


@app.route("/")
def home():
    return render_home()


@app.route("/api/models")
def api_models():
    """Whether the app's fixed model is installed on the Ollama server."""
    installed = classifier.list_installed_models()
    models = []
    for model in classifier.public_models():
        model["installed"] = (
            None
            if installed is None
            else classifier.normalise_tag(model["tag"]) in installed
        )
        models.append(model)
    return jsonify(server_online=installed is not None, models=models)


@app.route("/healthz")
def healthz():
    return jsonify(status="ok")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, generate_password_hash(password)),
                )
                db.commit()
                flash("Account created! You can now log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username or email is already registered.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("home"))
        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# --------------------------------------------------------------------------
# AI classification
# --------------------------------------------------------------------------
@app.route("/classify", methods=["GET", "POST"])
@login_required
def classify():
    result = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Please choose an image to upload.", "error")
        elif not allowed_file(file.filename):
            flash("That file type is not supported. Use JPG, PNG, WEBP, BMP, GIF, or AVIF.", "error")
        else:
            try:
                filename = secure_filename(file.filename)
                # Avoid name clashes between users
                path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    f"user{session['user_id']}_{filename}",
                )
                file.save(path)

                classifier.validate_image(path)
                pred = classifier.predict(path)
                category = pred["category"]
                result = {
                    "category": category,
                    "label": category.capitalize(),
                    "icon": classifier.ICONS.get(category, "♻"),
                    "verdict": pred["verdict"],
                    "verdict_label": classifier.VERDICT_LABELS[pred["verdict"]],
                    "verdict_class": pred["verdict"].replace("_", "-"),
                    "verdict_guidance": classifier.VERDICT_GUIDANCE[pred["verdict"]],
                    "advice": classifier.ADVICE[category],
                    "source": pred["source"],
                    "model_label": pred["model_label"],
                    "model_tag": pred["model_tag"],
                    "warning": pred.get("warning"),
                    "image": path,
                }
            except Exception as exc:  # noqa: BLE001 - show a friendly message
                flash(f"Sorry, I couldn't process that image: {exc}", "error")

    return render_home(result)


@app.context_processor
def inject_user():
    """Make the logged-in username available in every template."""
    return {"current_user": session.get("username")}


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug)
