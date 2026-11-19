from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


class FeatureEngineer:
    """构建用户特征、训练样本和推理样本。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "customer_state" not in self.df.columns:
            self.df["customer_state"] = "unknown"
        self.df["order_purchase_timestamp"] = pd.to_datetime(
            self.df["order_purchase_timestamp"], errors="coerce"
        )
        self.df = self.df.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])
        self.df["product_category_name"] = self.df["product_category_name"].fillna("unknown")

    @staticmethod
    def _safe_days(delta: pd.Timedelta) -> int:
        return max(int(delta.days), 0)

    @staticmethod
    def _recent_gap_days(order_times: pd.Series) -> float:
        if len(order_times) < 2:
            return 0.0
        gap = order_times.iloc[-1] - order_times.iloc[-2]
        return float(max(gap.days, 0))

    def _build_order_level(self, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
        source_df = self.df if source_df is None else source_df
        order_level = (
            source_df.groupby(["customer_unique_id", "order_id", "order_purchase_timestamp"], as_index=False)
            .agg(
                total_amount=("price", lambda s: float(s.fillna(0).sum())),
                total_freight=("freight_value", lambda s: float(s.fillna(0).sum())),
                total_items=("order_item_id", lambda s: float(s.notna().sum())),
                customer_state=("customer_state", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
                categories=("product_category_name", lambda s: list(s.fillna("unknown"))),
            )
            .sort_values(["customer_unique_id", "order_purchase_timestamp"])
            .reset_index(drop=True)
        )
        order_level["order_value"] = order_level["total_amount"] + order_level["total_freight"]
        return order_level

    def _build_sequence_and_static_features(
        self,
        history: pd.DataFrame,
        seq_length: int,
    ) -> tuple[list[list[float]], list[float]]:
        history = history.tail(seq_length).copy()
        first_history_time = history["order_purchase_timestamp"].min()

        seq_sample = []
        for _, row in history.iterrows():
            is_weekend = int(row["order_purchase_timestamp"].weekday() >= 5)
            hour = row["order_purchase_timestamp"].hour
            relative_days = self._safe_days(row["order_purchase_timestamp"] - first_history_time)
            seq_sample.append(
                [
                    float(row["order_value"]),
                    float(row["total_items"]),
                    float(relative_days),
                    float(is_weekend),
                    float(hour),
                ]
            )

        while len(seq_sample) < seq_length:
            seq_sample.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])

        order_values = history["order_value"].fillna(0)
        order_times = history["order_purchase_timestamp"]
        history_span_days = self._safe_days(order_times.max() - order_times.min())
        category_counts = Counter(cat for cats in history["categories"] for cat in cats)
        total_category_events = sum(category_counts.values())
        dominant_category_share = (
            float(category_counts.most_common(1)[0][1] / total_category_events) if total_category_events else 0.0
        )

        static_sample = [
            float(len(history)),
            float(order_values.sum()),
            float(order_values.mean()),
            float(order_values.std(ddof=0) if len(order_values) > 1 else 0.0),
            float(history["total_items"].sum()),
            float(history["total_items"].mean()),
            float(history_span_days),
            float(self._recent_gap_days(order_times)),
            float(history["order_purchase_timestamp"].dt.weekday.isin([5, 6]).mean()),
            float(history["order_purchase_timestamp"].dt.hour.mean()),
            dominant_category_share,
        ]
        return seq_sample, static_sample

    def create_user_features(self, reference_date: str | None = None) -> pd.DataFrame:
        if reference_date is None:
            reference_date = self.df["order_purchase_timestamp"].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        rows = []
        for user_id, group in self.df.groupby("customer_unique_id"):
            group = group.sort_values("order_purchase_timestamp")
            total_amount = (group["price"].fillna(0) + group["freight_value"].fillna(0)).sum()
            total_items = int(group["order_item_id"].notna().sum())
            purchase_span = self._safe_days(
                group["order_purchase_timestamp"].max() - group["order_purchase_timestamp"].min()
            )
            category_counts = group["product_category_name"].value_counts()
            rows.append(
                {
                    "user_id": user_id,
                    "total_orders": int(group["order_id"].nunique()),
                    "total_items": total_items,
                    "total_spent": float(total_amount),
                    "avg_order_value": float(total_amount / max(group["order_id"].nunique(), 1)),
                    "avg_items_per_order": float(total_items / max(group["order_id"].nunique(), 1)),
                    "first_purchase_days_ago": self._safe_days(
                        reference_date - group["order_purchase_timestamp"].min()
                    ),
                    "last_purchase_days_ago": self._safe_days(
                        reference_date - group["order_purchase_timestamp"].max()
                    ),
                    "purchase_span_days": purchase_span,
                    "orders_per_month": float(group["order_id"].nunique() / max(purchase_span / 30, 1)),
                    "customer_state": group["customer_state"].mode().iat[0],
                    "preferred_category": category_counts.index[0] if not category_counts.empty else "unknown",
                }
            )

        features = pd.DataFrame(rows)
        print(f"已为 {len(features)} 个用户构建画像特征。")
        return features

    def build_purchase_prediction_dataset(
        self,
        seq_length: int = 10,
        horizon_days: int = 30,
        min_history_orders: int = 1,
        top_k_categories: int = 20,
    ) -> dict[str, np.ndarray | list[str] | pd.DataFrame]:
        print("正在构建未来 30 天购买预测数据集...")
        order_level = self._build_order_level()

        top_categories = (
            self.df["product_category_name"].fillna("unknown").value_counts().head(top_k_categories).index.tolist()
        )
        category_to_idx = {name: idx for idx, name in enumerate(top_categories)}
        other_index = len(top_categories)
        category_names = top_categories + ["other"]

        sequence_features = []
        static_features = []
        labels_purchase = []
        labels_category = []
        sample_rows = []

        for user_id, group in order_level.groupby("customer_unique_id"):
            group = group.sort_values("order_purchase_timestamp").reset_index(drop=True)
            if len(group) <= min_history_orders:
                continue

            for i in range(min_history_orders, len(group)):
                history = group.iloc[:i].copy()
                current_time = history["order_purchase_timestamp"].max()
                future = group.iloc[i:].copy()
                horizon_end = current_time + pd.Timedelta(days=horizon_days)
                future_in_horizon = future[future["order_purchase_timestamp"] <= horizon_end]
                purchase_label = int(not future_in_horizon.empty)

                if purchase_label:
                    next_order = future_in_horizon.iloc[0]
                    next_category_counter = Counter(next_order["categories"])
                    next_category = (
                        next_category_counter.most_common(1)[0][0] if next_category_counter else "unknown"
                    )
                    category_label = category_to_idx.get(next_category, other_index)
                else:
                    category_label = other_index

                seq_sample, static_sample = self._build_sequence_and_static_features(
                    history=history,
                    seq_length=seq_length,
                )

                sequence_features.append(seq_sample)
                static_features.append(static_sample)
                labels_purchase.append(purchase_label)
                labels_category.append(category_label)
                sample_rows.append(
                    {
                        "user_id": user_id,
                        "snapshot_time": current_time,
                        "purchase_within_30d": purchase_label,
                        "target_category": category_names[category_label],
                        "history_orders": int(len(history)),
                    }
                )

        X_seq = np.asarray(sequence_features, dtype=np.float32)
        X_static = np.asarray(static_features, dtype=np.float32)
        y_purchase = np.asarray(labels_purchase, dtype=np.int64)
        y_category = np.asarray(labels_category, dtype=np.int64)
        sample_table = pd.DataFrame(sample_rows).sort_values("snapshot_time").reset_index(drop=True)

        print(f"样本构建完成，总样本数：{len(sample_table)}")
        print(f"正样本数量：{int(y_purchase.sum())}")
        if len(y_purchase):
            print(f"正样本占比：{y_purchase.mean():.4f}")

        return {
            "X_seq": X_seq,
            "X_static": X_static,
            "y_purchase": y_purchase,
            "y_category": y_category,
            "sample_table": sample_table,
            "category_names": category_names,
        }

    def build_latest_inference_dataset(
        self,
        category_names: list[str],
        seq_length: int = 10,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        print("正在构建最新用户推理数据集...")
        order_level = self._build_order_level()
        sequence_features = []
        static_features = []
        sample_rows = []

        for user_id, group in order_level.groupby("customer_unique_id"):
            group = group.sort_values("order_purchase_timestamp").reset_index(drop=True)
            if group.empty:
                continue

            seq_sample, static_sample = self._build_sequence_and_static_features(
                history=group,
                seq_length=seq_length,
            )
            sequence_features.append(seq_sample)
            static_features.append(static_sample)
            sample_rows.append(
                {
                    "user_id": user_id,
                    "snapshot_time": group["order_purchase_timestamp"].max(),
                    "history_orders": int(len(group)),
                }
            )

        return {
            "X_seq": np.asarray(sequence_features, dtype=np.float32),
            "X_static": np.asarray(static_features, dtype=np.float32),
            "sample_table": pd.DataFrame(sample_rows).sort_values("snapshot_time").reset_index(drop=True),
        }

    def build_single_user_inference_sample(
        self,
        user_id: str,
        category_names: list[str],
        seq_length: int = 10,
    ) -> dict[str, np.ndarray | dict[str, object]] | None:
        """只为单个用户构建推理样本，避免全量遍历。"""
        user_rows = self.df[self.df["customer_unique_id"].astype(str) == str(user_id)].copy()
        if user_rows.empty:
            return None

        order_level = self._build_order_level(user_rows)
        user_group = order_level.sort_values("order_purchase_timestamp").reset_index(drop=True)
        seq_sample, static_sample = self._build_sequence_and_static_features(
            history=user_group,
            seq_length=seq_length,
        )

        sample_info = {
            "user_id": user_id,
            "snapshot_time": user_group["order_purchase_timestamp"].max(),
            "history_orders": int(len(user_group)),
        }

        return {
            "X_seq": np.asarray([seq_sample], dtype=np.float32),
            "X_static": np.asarray([static_sample], dtype=np.float32),
            "sample_info": sample_info,
        }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path / "merged_data.csv")
    engineer = FeatureEngineer(df)

    user_features = engineer.create_user_features()
    user_features.to_csv(processed_path / "user_features.csv", index=False)

    dataset = engineer.build_purchase_prediction_dataset(seq_length=10, horizon_days=30)
    np.save(processed_path / "X_seq_purchase.npy", dataset["X_seq"])
    np.save(processed_path / "X_static_purchase.npy", dataset["X_static"])
    np.save(processed_path / "y_purchase.npy", dataset["y_purchase"])
    np.save(processed_path / "y_category.npy", dataset["y_category"])
    dataset["sample_table"].to_csv(processed_path / "purchase_samples.csv", index=False)
    pd.Series(dataset["category_names"], name="category_name").to_csv(
        processed_path / "category_labels.csv", index=False
    )

    print("特征工程完成。")
    print(f"序列特征形状：{dataset['X_seq'].shape}")
    print(f"静态特征形状：{dataset['X_static'].shape}")


if __name__ == "__main__":
    main()
