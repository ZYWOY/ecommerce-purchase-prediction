from __future__ import annotations

import os
import sqlite3
from functools import lru_cache, wraps
from pathlib import Path

import pandas as pd
from flask import Flask, Response, g, jsonify, render_template, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash

from src.inference import (
    load_feature_importance_report,
    predict_for_latest_users,
    predict_for_user_id,
    prepare_prediction_dataframe,
    project_root,
)


ROOT = project_root()
DATABASE_PATH = ROOT / "web" / "app.db"
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "merged_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "model_comparison.csv"
MODELS_DIR = ROOT / "models"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "ecommerce_prediction_secret")
    app.config["DATABASE"] = str(DATABASE_PATH)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db(app)

    @app.before_request
    def load_current_user() -> None:
        user_id = session.get("user_id")
        if user_id is None:
            g.user = None
            return

        db = get_db(app)
        g.user = db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    @app.teardown_appcontext
    def teardown_db(_error=None) -> None:
        close_db()

    @app.get("/")
    def index() -> str:
        return render_template("vue_index.html")

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"success": True, "message": "服务运行正常。"})

    @app.get("/api/session")
    def get_session_info():
        if g.user is None:
            return jsonify({"success": True, "authenticated": False, "user": None})

        return jsonify(
            {
                "success": True,
                "authenticated": True,
                "user": {
                    "id": g.user["id"],
                    "username": g.user["username"],
                    "created_at": g.user["created_at"],
                },
            }
        )

    @app.post("/api/register")
    def register():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        confirm_password = str(payload.get("confirm_password", ""))

        if not username or not password:
            return api_error("用户名和密码不能为空。", 400)
        if password != confirm_password:
            return api_error("两次输入的密码不一致。", 400)
        if len(password) < 6:
            return api_error("密码长度不能少于 6 位。", 400)

        db = get_db(app)
        existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return api_error("该用户名已存在，请更换后重试。", 400)

        db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        db.commit()
        return jsonify({"success": True, "message": "注册成功，请登录。"})

    @app.post("/api/login")
    def login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))

        if not username or not password:
            return api_error("请输入用户名和密码。", 400)

        db = get_db(app)
        user = db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            return api_error("用户名或密码错误。", 400)

        session.clear()
        session["user_id"] = user["id"]
        return jsonify(
            {
                "success": True,
                "message": "登录成功。",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "created_at": user["created_at"],
                },
            }
        )

    @app.post("/api/logout")
    def logout():
        session.clear()
        return jsonify({"success": True, "message": "已退出登录。"})

    @app.get("/api/dashboard")
    @api_login_required
    def dashboard():
        return jsonify(
            {
                "success": True,
                "summary": load_dashboard_summary(),
                "history": serialize_rows(get_prediction_history(app, g.user["id"])),
                "result_paths": {
                    "comparison_csv": str(RESULTS_PATH),
                    "database": str(DATABASE_PATH),
                    "processed_data": str(DEFAULT_DATA_PATH),
                    "models_dir": str(MODELS_DIR),
                },
            }
        )

    @app.get("/api/model-comparison")
    @api_login_required
    def model_comparison():
        if not RESULTS_PATH.exists():
            return api_error("未找到模型对比结果，请先运行训练脚本。", 404)

        comparison = pd.read_csv(RESULTS_PATH)
        numeric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "selection_score",
            "category_accuracy_positive_only",
            "category_top3_accuracy",
            "category_macro_f1",
            "category_weighted_f1",
        ]
        for column in numeric_columns:
            if column in comparison.columns:
                comparison[column] = pd.to_numeric(comparison[column], errors="coerce").fillna(0.0)

        return jsonify({"success": True, "models": comparison.to_dict(orient="records")})

    @app.get("/api/feature-importance")
    @api_login_required
    def feature_importance():
        importance = load_feature_importance_report(ROOT).head(12)
        return jsonify({"success": True, "features": importance.to_dict(orient="records")})

    @app.get("/api/artifacts")
    @api_login_required
    def artifacts():
        return jsonify({"success": True, "artifacts": list_result_artifacts()})

    @app.get("/results/<path:filename>")
    @api_login_required
    def serve_result_file(filename: str):
        safe_path = (RESULTS_DIR / filename).resolve()
        if RESULTS_DIR.resolve() not in safe_path.parents and safe_path != RESULTS_DIR.resolve():
            return api_error("非法文件路径。", 403)
        if not safe_path.exists():
            return api_error("请求的结果文件不存在。", 404)
        return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

    @app.post("/api/predict")
    @api_login_required
    def predict():
        payload = request.get_json(silent=True) or {}
        customer_unique_id = str(payload.get("customer_unique_id", "")).strip()
        if not customer_unique_id:
            return api_error("请输入用户 ID。", 400)

        try:
            df = load_default_data()
            result = predict_for_user_id(df, user_id=customer_unique_id, root=ROOT)
            if result is None:
                return api_error("未找到该用户的历史记录，请确认用户 ID 是否正确。", 404)

            save_prediction_record(app, g.user["id"], result)
            return jsonify({"success": True, "message": "预测完成。", "result": result})
        except ValueError as exc:
            return api_error(str(exc), 400)
        except FileNotFoundError as exc:
            return api_error(str(exc), 500)
        except Exception as exc:
            return api_error(f"预测过程中发生错误：{exc}", 500)

    @app.post("/api/batch-predict")
    @api_login_required
    def batch_predict():
        file = request.files.get("data_file")
        if file is None or file.filename == "":
            return api_error("请先选择 CSV 文件。", 400)

        try:
            df = prepare_prediction_dataframe(pd.read_csv(file))
            results = predict_for_latest_users(df, root=ROOT).head(50).to_dict(orient="records")
            return jsonify(
                {
                    "success": True,
                    "message": "批量预测完成，当前展示前 50 条结果。",
                    "filename": file.filename,
                    "results": results,
                }
            )
        except ValueError as exc:
            return api_error(str(exc), 400)
        except FileNotFoundError as exc:
            return api_error(str(exc), 500)
        except Exception as exc:
            return api_error(f"批量预测失败：{exc}", 500)

    return app


def api_login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return api_error("请先登录后再访问该接口。", 401)
        return view(*args, **kwargs)

    return wrapped_view


def api_error(message: str, status_code: int = 400) -> tuple[Response, int]:
    return jsonify({"success": False, "message": message}), status_code


def get_db(app: Flask) -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db() -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app: Flask) -> None:
    connection = sqlite3.connect(app.config["DATABASE"])
    cursor = connection.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            customer_unique_id TEXT NOT NULL,
            purchase_probability REAL NOT NULL,
            predicted_purchase INTEGER NOT NULL,
            recommended_category TEXT NOT NULL,
            model_name TEXT NOT NULL,
            history_orders INTEGER NOT NULL,
            snapshot_time TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    )
    connection.commit()
    connection.close()


@lru_cache(maxsize=1)
def load_default_data() -> pd.DataFrame:
    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(f"未找到默认数据文件：{DEFAULT_DATA_PATH}")
    return pd.read_csv(DEFAULT_DATA_PATH)


def load_dashboard_summary() -> dict[str, object]:
    summary = {
        "record_count": 0,
        "user_count": 0,
        "best_model": "-",
        "best_f1": 0.0,
        "top_categories": {},
        "history_count": 0,
    }

    if DEFAULT_DATA_PATH.exists():
        df = load_default_data()
        summary["record_count"] = len(df)
        summary["user_count"] = int(df["customer_unique_id"].nunique())

    if RESULTS_PATH.exists():
        comparison = pd.read_csv(RESULTS_PATH)
        sort_column = None
        if not comparison.empty and {"model", "selection_score"}.issubset(comparison.columns):
            sort_column = "selection_score"
        elif not comparison.empty and {"model", "f1"}.issubset(comparison.columns):
            sort_column = "f1"

        if sort_column is not None:
            comparison = comparison.sort_values(sort_column, ascending=False)
            best_row = comparison.iloc[0]
            summary["best_model"] = str(best_row["model"])
            summary["best_f1"] = float(best_row["f1"])

    category_path = ROOT / "data" / "processed" / "purchase_samples.csv"
    if category_path.exists():
        samples = pd.read_csv(category_path, usecols=["target_category"])
        summary["top_categories"] = samples["target_category"].value_counts().head(5).to_dict()

    if DATABASE_PATH.exists():
        connection = sqlite3.connect(DATABASE_PATH)
        try:
            summary["history_count"] = int(connection.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0])
        finally:
            connection.close()

    return summary


def list_result_artifacts() -> dict[str, object]:
    models = ["lstm", "gru", "mlp_fusion", "transformer"]
    image_groups = []
    metric_files = []
    prediction_files = []

    for model_name in models:
        image_groups.append(
            {
                "model": model_name,
                "training_history": build_result_file_entry(f"{model_name}_training_history.png", "训练曲线"),
                "roc_curve": build_result_file_entry(f"{model_name}_roc_curve.png", "ROC 曲线"),
                "confusion_matrix": build_result_file_entry(f"{model_name}_confusion_matrix.png", "混淆矩阵"),
                "permutation_importance": build_result_file_entry(
                    f"{model_name}_permutation_importance.png", "Permutation Importance"
                ),
            }
        )
        metric_files.append(build_result_file_entry(f"{model_name}_metrics.json", f"{model_name} 指标文件"))
        prediction_files.append(
            build_result_file_entry(f"{model_name}_test_predictions.csv", f"{model_name} 测试集预测结果")
        )
        prediction_files.append(
            build_result_file_entry(
                f"{model_name}_permutation_importance.csv", f"{model_name} Permutation Importance"
            )
        )

    return {
        "comparison": build_result_file_entry("model_comparison.csv", "模型对比总表"),
        "metric_files": metric_files,
        "prediction_files": prediction_files,
        "image_groups": image_groups,
    }


def build_result_file_entry(filename: str, label: str) -> dict[str, str | bool]:
    path = RESULTS_DIR / filename
    return {
        "label": label,
        "filename": filename,
        "path": str(path),
        "url": f"/results/{filename}",
        "exists": path.exists(),
    }


def save_prediction_record(app: Flask, user_pk: int, result: dict[str, object]) -> None:
    db = get_db(app)
    db.execute(
        """
        INSERT INTO prediction_history (
            user_id, customer_unique_id, purchase_probability, predicted_purchase,
            recommended_category, model_name, history_orders, snapshot_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_pk,
            str(result["user_id"]),
            float(result["purchase_probability"]),
            int(result["predicted_purchase"]),
            str(result.get("recommended_category") or ""),
            str(result["model_name"]),
            int(result["history_orders"]),
            str(result["snapshot_time"]),
        ),
    )
    db.commit()


def get_prediction_history(app: Flask, user_pk: int):
    db = get_db(app)
    return db.execute(
        """
        SELECT customer_unique_id, purchase_probability, predicted_purchase,
               recommended_category, model_name, history_orders, snapshot_time, created_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (user_pk,),
    ).fetchall()


def serialize_rows(rows) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
