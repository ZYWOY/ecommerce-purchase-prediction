from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .feature_engineering import FeatureEngineer
    from .feature_scaling import load_feature_scaler, transform_feature_arrays
    from .models import MODEL_REGISTRY
except ImportError:
    from feature_engineering import FeatureEngineer
    from feature_scaling import load_feature_scaler, transform_feature_arrays
    from models import MODEL_REGISTRY


REQUIRED_PREDICTION_COLUMNS = {
    "customer_unique_id",
    "order_id",
    "order_item_id",
    "order_purchase_timestamp",
    "price",
    "freight_value",
    "product_category_name",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_category_names(root: Path | None = None) -> list[str]:
    root = root or project_root()
    return pd.read_csv(root / "data" / "processed" / "category_labels.csv")["category_name"].tolist()


def load_best_model_name(root: Path | None = None) -> str:
    root = root or project_root()
    comparison_path = root / "results" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        sort_column = None
        if {"model", "selection_score"}.issubset(comparison.columns):
            sort_column = "selection_score"
        elif {"model", "f1"}.issubset(comparison.columns):
            sort_column = "f1"

        if sort_column is not None:
            comparison = comparison.sort_values(sort_column, ascending=False)
            return str(comparison.iloc[0]["model"])

    for name in ["lstm", "gru", "mlp_fusion", "transformer"]:
        if (root / "models" / f"best_{name}.pth").exists():
            return name
    return "lstm"


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "customer_state" not in normalized.columns:
        normalized["customer_state"] = "unknown"

    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(normalized.columns))
    if missing_columns:
        raise ValueError(f"缺少必要字段：{', '.join(missing_columns)}")
    if normalized.empty:
        raise ValueError("上传的数据为空，无法执行预测。")
    return normalized


def load_purchase_threshold(model_name: str, root: Path | None = None) -> float:
    root = root or project_root()
    config_path = root / "models" / f"best_{model_name}_config.json"
    if not config_path.exists():
        return 0.5

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.5
    return float(config.get("purchase_threshold", 0.5))


def load_saved_feature_scaler(root: Path | None = None) -> dict[str, np.ndarray]:
    root = root or project_root()
    scaler_path = root / "models" / "feature_scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"未找到特征缩放文件：{scaler_path}。请先重新运行 src/train.py。")
    return load_feature_scaler(scaler_path)


def _model_cache_key(model_name: str | None, root: Path | None, device: str | None) -> tuple[str, str, str]:
    resolved_root = str((root or project_root()).resolve())
    resolved_model_name = model_name or load_best_model_name(Path(resolved_root))
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return resolved_model_name, resolved_root, resolved_device


@lru_cache(maxsize=6)
def _load_trained_model_cached(model_name: str, root_str: str, device: str):
    root = Path(root_str)
    category_names = load_category_names(root)
    X_seq = np.load(root / "data" / "processed" / "X_seq_purchase.npy")
    X_static = np.load(root / "data" / "processed" / "X_static_purchase.npy")

    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {
        "seq_input_size": X_seq.shape[-1],
        "static_input_size": X_static.shape[1],
        "num_categories": len(category_names),
    }
    if model_name in {"mlp_fusion", "transformer"}:
        kwargs["seq_length"] = X_seq.shape[1]

    model = model_cls(**kwargs).to(device)
    state_path = root / "models" / f"best_{model_name}.pth"
    if not state_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{state_path}。请先运行 src/train.py 完成模型训练。")

    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model, category_names


def load_trained_model(model_name: str | None = None, root: Path | None = None, device: str | None = None):
    resolved_model_name, resolved_root, resolved_device = _model_cache_key(model_name, root, device)
    model, category_names = _load_trained_model_cached(resolved_model_name, resolved_root, resolved_device)
    threshold = load_purchase_threshold(resolved_model_name, Path(resolved_root))
    return model, resolved_model_name, category_names, resolved_device, threshold


@lru_cache(maxsize=2)
def _load_seq_length(root_str: str) -> int:
    root = Path(root_str)
    return int(np.load(root / "data" / "processed" / "X_seq_purchase.npy").shape[1])


def build_prediction_dataset(df: pd.DataFrame, root: Path | None = None) -> dict[str, object]:
    root = root or project_root()
    category_names = load_category_names(root)
    seq_length = _load_seq_length(str(root.resolve()))
    engineer = FeatureEngineer(prepare_prediction_dataframe(df))
    dataset = engineer.build_latest_inference_dataset(category_names=category_names, seq_length=seq_length)
    scaler = load_saved_feature_scaler(root)
    X_seq, X_static = transform_feature_arrays(dataset["X_seq"], dataset["X_static"], scaler)
    dataset["X_seq"] = X_seq
    dataset["X_static"] = X_static
    return dataset


def predict_for_latest_users(
    df: pd.DataFrame,
    model_name: str | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    root = root or project_root()
    model, resolved_model_name, category_names, device, threshold = load_trained_model(model_name=model_name, root=root)
    dataset = build_prediction_dataset(df, root=root)

    seq_x = torch.tensor(dataset["X_seq"], dtype=torch.float32).to(device)
    static_x = torch.tensor(dataset["X_static"], dtype=torch.float32).to(device)

    with torch.no_grad():
        purchase_logits, category_logits = model(seq_x, static_x)
        purchase_prob = torch.sigmoid(purchase_logits).cpu().numpy()
        category_idx = torch.argmax(category_logits, dim=1).cpu().numpy()

    result = dataset["sample_table"].copy()
    result["purchase_probability"] = purchase_prob
    result["predicted_purchase"] = (purchase_prob >= threshold).astype(int)
    result["recommended_category"] = [
        category_names[idx] if predicted_purchase == 1 else None
        for idx, predicted_purchase in zip(category_idx, result["predicted_purchase"])
    ]
    result["model_name"] = resolved_model_name
    result["purchase_threshold"] = threshold
    return result.sort_values("purchase_probability", ascending=False).reset_index(drop=True)


def predict_for_user_id(
    df: pd.DataFrame,
    user_id: str,
    model_name: str | None = None,
    root: Path | None = None,
) -> dict[str, object] | None:
    root = root or project_root()
    model, resolved_model_name, category_names, device, threshold = load_trained_model(model_name=model_name, root=root)

    engineer = FeatureEngineer(prepare_prediction_dataframe(df))
    single_sample = engineer.build_single_user_inference_sample(
        user_id=user_id,
        category_names=category_names,
        seq_length=_load_seq_length(str(root.resolve())),
    )
    if single_sample is None:
        return None

    scaler = load_saved_feature_scaler(root)
    X_seq, X_static = transform_feature_arrays(single_sample["X_seq"], single_sample["X_static"], scaler)
    seq_x = torch.tensor(X_seq, dtype=torch.float32).to(device)
    static_x = torch.tensor(X_static, dtype=torch.float32).to(device)

    with torch.no_grad():
        purchase_logits, category_logits = model(seq_x, static_x)
        purchase_prob = float(torch.sigmoid(purchase_logits)[0].cpu().item())
        category_idx = int(torch.argmax(category_logits, dim=1)[0].cpu().item())

    sample_info = single_sample["sample_info"]
    predicted_purchase = int(purchase_prob >= threshold)
    return {
        "user_id": sample_info["user_id"],
        "snapshot_time": str(sample_info["snapshot_time"]),
        "history_orders": int(sample_info["history_orders"]),
        "purchase_probability": purchase_prob,
        "predicted_purchase": predicted_purchase,
        "recommended_category": category_names[category_idx] if predicted_purchase == 1 else None,
        "model_name": resolved_model_name,
        "purchase_threshold": threshold,
    }


def compute_feature_importance_proxy(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    X_static = np.load(root / "data" / "processed" / "X_static_purchase.npy")
    y_purchase = np.load(root / "data" / "processed" / "y_purchase.npy")
    feature_names = [
        "history_orders",
        "total_spent",
        "avg_order_value",
        "order_value_std",
        "total_items",
        "avg_items_per_order",
        "history_span_days",
        "recent_gap_days",
        "weekend_ratio",
        "avg_purchase_hour",
        "dominant_category_share",
    ]

    rows = []
    for idx, name in enumerate(feature_names):
        series = X_static[:, idx]
        if np.std(series) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(series, y_purchase)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        rows.append({"feature": name, "importance_score": abs(corr), "direction": np.sign(corr)})

    return pd.DataFrame(rows).sort_values("importance_score", ascending=False).reset_index(drop=True)


def load_feature_importance_report(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    model_name = load_best_model_name(root)
    importance_path = root / "results" / f"{model_name}_permutation_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        if {"feature", "importance"}.issubset(importance_df.columns):
            return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return compute_feature_importance_proxy(root)
