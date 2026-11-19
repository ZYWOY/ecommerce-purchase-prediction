from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """加载、清洗并合并 Olist 原始数据表。"""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.customers: pd.DataFrame | None = None
        self.orders: pd.DataFrame | None = None
        self.order_items: pd.DataFrame | None = None
        self.products: pd.DataFrame | None = None
        self.reviews: pd.DataFrame | None = None
        self.full_data: pd.DataFrame | None = None

    def load_data(self) -> "DataPreprocessor":
        print("正在加载原始数据集...")
        self.customers = pd.read_csv(self.data_path / "olist_customers_dataset.csv")
        self.orders = pd.read_csv(self.data_path / "olist_orders_dataset.csv")
        self.order_items = pd.read_csv(self.data_path / "olist_order_items_dataset.csv")
        self.products = pd.read_csv(self.data_path / "olist_products_dataset.csv")
        self.reviews = pd.read_csv(self.data_path / "olist_order_reviews_dataset.csv")
        print("原始数据集加载完成。")
        return self

    def clean_data(self) -> "DataPreprocessor":
        print("正在清洗数据...")
        if self.orders is None or self.products is None:
            raise ValueError("请先调用 load_data()，再执行 clean_data()。")

        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for col in date_cols:
            self.orders[col] = pd.to_datetime(self.orders[col], errors="coerce")

        self.orders = self.orders[self.orders["order_status"] == "delivered"].copy()
        self.orders = self.orders.dropna(
            subset=["order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date"]
        )

        self.products = self.products.copy()
        self.products["product_category_name"] = self.products["product_category_name"].fillna("unknown")

        for col in ["product_name_lenght", "product_description_lenght", "product_photos_qty"]:
            if col in self.products.columns:
                self.products[col] = self.products[col].fillna(self.products[col].median())

        print(f"清洗后保留的已完成订单数：{len(self.orders)}")
        print(f"清洗后保留的用户数：{self.orders['customer_id'].nunique()}")
        return self

    def merge_data(self) -> pd.DataFrame:
        print("正在合并数据表...")
        if any(df is None for df in [self.orders, self.order_items, self.products, self.customers]):
            raise ValueError("请先依次执行 load_data() 和 clean_data()，再执行 merge_data()。")

        orders_items = self.orders.merge(self.order_items, on="order_id", how="left")
        orders_items = orders_items.merge(
            self.products[
                [
                    "product_id",
                    "product_category_name",
                    "product_name_lenght",
                    "product_description_lenght",
                    "product_photos_qty",
                    "product_weight_g",
                    "product_length_cm",
                    "product_height_cm",
                    "product_width_cm",
                ]
            ],
            on="product_id",
            how="left",
        )
        self.full_data = orders_items.merge(
            self.customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id",
            how="left",
        )
        self.full_data["product_category_name"] = self.full_data["product_category_name"].fillna("unknown")
        self.full_data = self.full_data.sort_values("order_purchase_timestamp").reset_index(drop=True)

        print(f"合并后数据维度：{self.full_data.shape}")
        return self.full_data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor(raw_path)
    preprocessor.load_data().clean_data()
    df = preprocessor.merge_data()

    output_file = processed_path / "merged_data.csv"
    df.to_csv(output_file, index=False)
    print(f"数据预处理完成，合并数据已保存到：{output_file}")


if __name__ == "__main__":
    main()
