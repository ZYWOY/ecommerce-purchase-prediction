from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

try:
    from .feature_scaling import fit_feature_scaler, save_feature_scaler, transform_feature_arrays
    from .models import MODEL_REGISTRY
except ImportError:
    from feature_scaling import fit_feature_scaler, save_feature_scaler, transform_feature_arrays
    from models import MODEL_REGISTRY


STATIC_FEATURE_NAMES = [
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

SEQUENCE_FEATURE_NAMES = [
    "seq_order_value",
    "seq_total_items",
    "seq_relative_days",
    "seq_is_weekend",
    "seq_hour",
]


class Trainer:
    def __init__(self, model: nn.Module, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auc": [],
        }
        self.best_purchase_threshold = 0.5
        self.project_root = Path(__file__).resolve().parents[1]
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.scaler_path = self.models_dir / "feature_scaler.npz"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        y_purchase: np.ndarray,
        y_category: np.ndarray,
        sample_table: pd.DataFrame,
        batch_size: int = 64,
    ) -> None:
        sorted_index = sample_table.sort_values("snapshot_time").index.to_numpy()
        X_seq = X_seq[sorted_index]
        X_static = X_static[sorted_index]
        y_purchase = y_purchase[sorted_index]
        y_category = y_category[sorted_index]

        total = len(sample_table)
        train_end = max(int(total * 0.7), 1)
        val_end = max(int(total * 0.9), train_end + 1)

        splits = {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, total),
        }

        scaler = fit_feature_scaler(
            X_seq[splits["train"]],
            X_static[splits["train"]],
        )
        save_feature_scaler(self.scaler_path, scaler)
        X_seq, X_static = transform_feature_arrays(X_seq, X_static, scaler)

        self.train_loader = self._build_loader(X_seq, X_static, y_purchase, y_category, splits["train"], batch_size, True)
        self.val_loader = self._build_loader(X_seq, X_static, y_purchase, y_category, splits["val"], batch_size, False)
        self.test_loader = self._build_loader(X_seq, X_static, y_purchase, y_category, splits["test"], batch_size, False)
        self.test_targets = {
            "purchase": y_purchase[splits["test"]],
            "category": y_category[splits["test"]],
        }

        print(f"训练集样本数：{len(self.train_loader.dataset)}")
        print(f"验证集样本数：{len(self.val_loader.dataset)}")
        print(f"测试集样本数：{len(self.test_loader.dataset)}")

    @staticmethod
    def _build_loader(
        X_seq: np.ndarray,
        X_static: np.ndarray,
        y_purchase: np.ndarray,
        y_category: np.ndarray,
        split_slice: slice,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X_seq[split_slice], dtype=torch.float32),
            torch.tensor(X_static[split_slice], dtype=torch.float32),
            torch.tensor(y_purchase[split_slice], dtype=torch.float32),
            torch.tensor(y_category[split_slice], dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer | None,
        purchase_criterion: nn.Module,
        category_criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:
        train_mode = optimizer is not None
        self.model.train(train_mode)

        total_loss = 0.0
        all_probs, all_labels = [], []

        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for seq_x, static_x, y_purchase, y_category in loader:
                seq_x = seq_x.to(self.device)
                static_x = static_x.to(self.device)
                y_purchase = y_purchase.to(self.device)
                y_category = y_category.to(self.device)

                if train_mode:
                    optimizer.zero_grad()

                purchase_logits, category_logits = self.model(seq_x, static_x)
                purchase_loss = purchase_criterion(purchase_logits, y_purchase)
                positive_mask = y_purchase > 0.5
                if positive_mask.any():
                    category_loss = category_criterion(category_logits[positive_mask], y_category[positive_mask])
                    loss = purchase_loss + 0.3 * category_loss
                else:
                    loss = purchase_loss

                if train_mode:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                all_probs.extend(torch.sigmoid(purchase_logits).detach().cpu().numpy())
                all_labels.extend(y_purchase.detach().cpu().numpy())

        metrics = self._classification_metrics(np.asarray(all_labels), np.asarray(all_probs))
        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, metrics

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = float("nan")
        return metrics

    @staticmethod
    def _selection_score(metrics: dict[str, float]) -> float:
        return 0.5 * float(metrics["accuracy"]) + 0.5 * float(metrics["f1"])

    def fit(self, epochs: int = 30, lr: float = 1e-3, patience: int = 6) -> dict[str, list[float]]:
        purchase_criterion = nn.BCEWithLogitsLoss()
        category_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_val_score = -1.0
        patience_counter = 0
        best_path = self.models_dir / f"best_{self.model_name}.pth"

        print(f"开始在 {self.device} 设备上训练 {self.model_name} 模型...")
        for epoch in range(1, epochs + 1):
            train_loss, _ = self._run_epoch(self.train_loader, optimizer, purchase_criterion, category_criterion)
            val_loss, val_metrics = self._run_epoch(self.val_loader, None, purchase_criterion, category_criterion)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_auc"].append(val_metrics["roc_auc"])
            val_score = self._selection_score(val_metrics)
            scheduler.step(val_score)

            print(
                f"第 {epoch:02d} 轮 | "
                f"训练损失={train_loss:.4f} | 验证损失={val_loss:.4f} | "
                f"验证准确率={val_metrics['accuracy']:.4f} | 验证F1={val_metrics['f1']:.4f} | "
                f"验证AUC={val_metrics['roc_auc']:.4f}"
            )

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"在第 {epoch} 轮触发早停。")
                    break

        return self.history

    @staticmethod
    def _select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) <= 1:
            return 0.5

        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.1, 0.9, 81):
            y_pred = (y_prob >= threshold).astype(int)
            score = 0.5 * accuracy_score(y_true, y_pred) + 0.5 * f1_score(y_true, y_pred, zero_division=0)
            if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = float(threshold)
        return best_threshold

    def _predict_loader(self, loader: DataLoader) -> dict[str, np.ndarray]:
        purchase_probs = []
        category_probs = []
        category_preds = []
        purchase_targets = []
        category_targets = []

        self.model.eval()
        with torch.no_grad():
            for seq_x, static_x, y_purchase, y_category in loader:
                seq_x = seq_x.to(self.device)
                static_x = static_x.to(self.device)
                purchase_logits, category_logits = self.model(seq_x, static_x)
                probs = torch.sigmoid(purchase_logits).cpu().numpy()
                cat_probs = torch.softmax(category_logits, dim=1).cpu().numpy()
                purchase_probs.extend(probs)
                category_probs.extend(cat_probs)
                category_preds.extend(torch.argmax(category_logits, dim=1).cpu().numpy())
                purchase_targets.extend(y_purchase.numpy())
                category_targets.extend(y_category.numpy())

        return {
            "purchase_probs": np.asarray(purchase_probs),
            "category_probs": np.asarray(category_probs),
            "category_preds": np.asarray(category_preds),
            "purchase_targets": np.asarray(purchase_targets),
            "category_targets": np.asarray(category_targets),
        }

    def evaluate(self, category_names: list[str]) -> dict[str, float]:
        best_path = self.models_dir / f"best_{self.model_name}.pth"
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        val_outputs = self._predict_loader(self.val_loader)
        self.best_purchase_threshold = self._select_best_threshold(
            val_outputs["purchase_targets"],
            val_outputs["purchase_probs"],
        )

        test_outputs = self._predict_loader(self.test_loader)
        y_true = test_outputs["purchase_targets"]
        purchase_probs = test_outputs["purchase_probs"]
        purchase_preds = (purchase_probs >= self.best_purchase_threshold).astype(int)
        category_probs = test_outputs["category_probs"]
        category_preds = test_outputs["category_preds"]

        metrics = self._classification_metrics(y_true, purchase_probs, threshold=self.best_purchase_threshold)
        metrics["purchase_threshold"] = self.best_purchase_threshold
        metrics["selection_score"] = self._selection_score(metrics)
        category_mask = y_true == 1
        if category_mask.sum() > 0:
            true_categories = test_outputs["category_targets"][category_mask]
            pred_categories = category_preds[category_mask]
            positive_category_probs = category_probs[category_mask]
            metrics["category_accuracy_positive_only"] = accuracy_score(
                true_categories,
                pred_categories,
            )
            metrics["category_top3_accuracy"] = self._top_k_category_accuracy(
                true_categories,
                positive_category_probs,
                k=min(3, positive_category_probs.shape[1]),
            )
            metrics["category_macro_f1"] = f1_score(
                true_categories,
                pred_categories,
                average="macro",
                zero_division=0,
            )
            metrics["category_weighted_f1"] = f1_score(
                true_categories,
                pred_categories,
                average="weighted",
                zero_division=0,
            )
        else:
            metrics["category_accuracy_positive_only"] = float("nan")
            metrics["category_top3_accuracy"] = float("nan")
            metrics["category_macro_f1"] = float("nan")
            metrics["category_weighted_f1"] = float("nan")

        self._save_confusion_matrix(y_true, np.asarray(purchase_preds))
        self._save_roc_curve(y_true, np.asarray(purchase_probs))
        self._save_permutation_importance()

        metrics_path = self.results_dir / f"{self.model_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(self.models_dir / f"best_{self.model_name}_config.json", "w", encoding="utf-8") as f:
            json.dump({"purchase_threshold": self.best_purchase_threshold}, f, ensure_ascii=False, indent=2)

        prediction_preview = pd.DataFrame(
            {
                "真实购买标签": y_true,
                "预测购买标签": purchase_preds,
                "购买概率": purchase_probs,
                "预测商品类别": [category_names[idx] for idx in category_preds],
            }
        )
        prediction_preview.to_csv(self.results_dir / f"{self.model_name}_test_predictions.csv", index=False)
        return metrics

    @staticmethod
    def _top_k_category_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
        top_k = np.argsort(y_prob, axis=1)[:, -k:]
        return float(np.mean([label in row for label, row in zip(y_true, top_k)]))

    def _collect_test_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_batches = []
        static_batches = []
        label_batches = []
        for seq_x, static_x, y_purchase, _ in self.test_loader:
            seq_batches.append(seq_x.numpy())
            static_batches.append(static_x.numpy())
            label_batches.append(y_purchase.numpy())
        return np.concatenate(seq_batches), np.concatenate(static_batches), np.concatenate(label_batches)

    def _predict_purchase_probabilities(self, X_seq: np.ndarray, X_static: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(X_seq), batch_size):
                seq_x = torch.tensor(X_seq[start : start + batch_size], dtype=torch.float32).to(self.device)
                static_x = torch.tensor(X_static[start : start + batch_size], dtype=torch.float32).to(self.device)
                purchase_logits, _ = self.model(seq_x, static_x)
                probs.extend(torch.sigmoid(purchase_logits).cpu().numpy())
        return np.asarray(probs)

    def _save_permutation_importance(self) -> None:
        X_seq, X_static, y_true = self._collect_test_arrays()
        baseline_probs = self._predict_purchase_probabilities(X_seq, X_static)
        baseline_f1 = self._classification_metrics(y_true, baseline_probs, threshold=self.best_purchase_threshold)["f1"]
        rng = np.random.default_rng(42)
        rows = []

        for idx, name in enumerate(SEQUENCE_FEATURE_NAMES):
            permuted_seq = X_seq.copy()
            flat_values = permuted_seq[:, :, idx].reshape(-1)
            permuted_values = rng.permutation(flat_values)
            permuted_seq[:, :, idx] = permuted_values.reshape(permuted_seq[:, :, idx].shape)
            permuted_probs = self._predict_purchase_probabilities(permuted_seq, X_static)
            permuted_f1 = self._classification_metrics(
                y_true,
                permuted_probs,
                threshold=self.best_purchase_threshold,
            )["f1"]
            rows.append(
                {
                    "feature": name,
                    "feature_type": "sequence",
                    "baseline_f1": baseline_f1,
                    "permuted_f1": permuted_f1,
                    "importance": max(baseline_f1 - permuted_f1, 0.0),
                }
            )

        for idx, name in enumerate(STATIC_FEATURE_NAMES):
            permuted_static = X_static.copy()
            permuted_static[:, idx] = rng.permutation(permuted_static[:, idx])
            permuted_probs = self._predict_purchase_probabilities(X_seq, permuted_static)
            permuted_f1 = self._classification_metrics(
                y_true,
                permuted_probs,
                threshold=self.best_purchase_threshold,
            )["f1"]
            rows.append(
                {
                    "feature": name,
                    "feature_type": "static",
                    "baseline_f1": baseline_f1,
                    "permuted_f1": permuted_f1,
                    "importance": max(baseline_f1 - permuted_f1, 0.0),
                }
            )

        importance_df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(self.results_dir / f"{self.model_name}_permutation_importance.csv", index=False)

        top_features = importance_df.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_features["feature"], top_features["importance"], color="#1f6d63")
        ax.set_title(f"{self.model_name.upper()} Permutation Importance")
        ax.set_xlabel("F1 drop after permutation")
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.model_name}_permutation_importance.png", dpi=200)
        plt.close(fig)

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{self.model_name.upper()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.model_name}_confusion_matrix.png", dpi=200)
        plt.close(fig)

    def _save_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        if len(np.unique(y_true)) <= 1:
            return
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(f"{self.model_name.upper()} ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.model_name}_roc_curve.png", dpi=200)
        plt.close(fig)

    def plot_history(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"], label="Validation Loss")
        axes[0].set_title(f"{self.model_name.upper()} Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(self.history["val_f1"], label="Validation F1", color="green")
        axes[1].plot(self.history["val_auc"], label="Validation AUC", color="orange")
        axes[1].set_title(f"{self.model_name.upper()} Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.model_name}_training_history.png", dpi=200)
        plt.close(fig)


def load_processed_dataset(project_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    processed = project_root / "data" / "processed"
    X_seq = np.load(processed / "X_seq_purchase.npy")
    X_static = np.load(processed / "X_static_purchase.npy")
    y_purchase = np.load(processed / "y_purchase.npy")
    y_category = np.load(processed / "y_category.npy")
    sample_table = pd.read_csv(processed / "purchase_samples.csv")
    category_names = pd.read_csv(processed / "category_labels.csv")["category_name"].tolist()
    return X_seq, X_static, y_purchase, y_category, sample_table, category_names


def run_experiment(model_key: str) -> dict[str, float]:
    project_root = Path(__file__).resolve().parents[1]
    X_seq, X_static, y_purchase, y_category, sample_table, category_names = load_processed_dataset(project_root)

    model_cls = MODEL_REGISTRY[model_key]
    model_kwargs = {
        "seq_input_size": X_seq.shape[-1],
        "static_input_size": X_static.shape[1],
        "num_categories": len(category_names),
    }
    if model_key in {"mlp_fusion", "transformer"}:
        model_kwargs["seq_length"] = X_seq.shape[1]

    model = model_cls(**model_kwargs)
    trainer = Trainer(model=model, model_name=model_key)
    trainer.prepare_data(X_seq, X_static, y_purchase, y_category, sample_table)
    trainer.fit()
    metrics = trainer.evaluate(category_names)
    trainer.plot_history()

    print(f"{model_key} 模型测试集指标：")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return metrics


def main() -> None:
    summary = {}
    for model_key in MODEL_REGISTRY:
        summary[model_key] = run_experiment(model_key)

    summary_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "model"})
    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "results" / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"模型对比结果已保存到：{summary_path}")


if __name__ == "__main__":
    main()
