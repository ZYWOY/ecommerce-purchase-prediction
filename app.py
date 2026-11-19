from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference import (
    load_feature_importance_report,
    load_best_model_name,
    prepare_prediction_dataframe,
    predict_for_latest_users,
    project_root,
)


ROOT = project_root()
PROCESSED_DATA = ROOT / "data" / "processed" / "merged_data.csv"
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="电商购买预测系统",
    page_icon="📦",
    layout="wide",
)

st.title("电商用户购买行为预测系统")
st.caption("基于 Olist 数据集的未来 30 天购买预测与商品类别推荐备用演示页面")


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        return pd.read_csv(comparison_path)
    return pd.DataFrame()


uploaded = st.file_uploader("上传用户行为数据（CSV）", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    data_source = "上传文件"
else:
    df = load_default_data()
    data_source = "默认处理后数据"

try:
    df = prepare_prediction_dataframe(df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("系统信息")
st.sidebar.write(f"数据来源：{data_source}")
st.sidebar.write(f"当前最佳模型：`{load_best_model_name(ROOT)}`")
st.sidebar.write(f"记录数：{len(df):,}")
st.sidebar.write(f"用户数：{df['customer_unique_id'].nunique():,}")

predictions = predict_for_latest_users(df, root=ROOT)
importance_df = load_feature_importance_report(ROOT)
comparison_df = load_model_comparison()

col1, col2, col3, col4 = st.columns(4)
col1.metric("预测样本数", f"{len(predictions):,}")
col2.metric("预测购买用户占比", f"{(predictions['predicted_purchase'].mean() * 100):.2f}%")
col3.metric("平均购买概率", f"{(predictions['purchase_probability'].mean() * 100):.2f}%")
col4.metric("推荐类别数", f"{predictions['recommended_category'].dropna().nunique():,}")

st.subheader("购买概率分布")
distribution = (
    predictions["purchase_probability"]
    .clip(0, 0.9999)
    .mul(10)
    .astype(int)
    .value_counts()
    .sort_index()
)
distribution.index = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in distribution.index]
st.bar_chart(distribution)

if not comparison_df.empty:
    st.subheader("模型对比结果")
    st.dataframe(comparison_df, use_container_width=True)

st.subheader("推荐商品类别 Top 10")
recommended_categories = predictions["recommended_category"].dropna()
if recommended_categories.empty:
    st.info("当前预测结果中暂无可展示的推荐类别。")
else:
    st.bar_chart(recommended_categories.value_counts().head(10))

st.subheader("特征重要性代理排序")
st.dataframe(importance_df, use_container_width=True)

st.subheader("单用户查询")
selected_user = st.selectbox("选择用户 ID", options=predictions["user_id"].tolist())
user_row = predictions[predictions["user_id"] == selected_user].iloc[0]
st.write(
    {
        "用户ID": user_row["user_id"],
        "样本时间": str(user_row["snapshot_time"]),
        "历史订单数": int(user_row["history_orders"]),
        "购买概率": f"{user_row['purchase_probability']:.4f}",
        "是否预测购买": "是" if int(user_row["predicted_purchase"]) == 1 else "否",
        "推荐商品类别": user_row["recommended_category"] or "暂无推荐",
        "模型名称": user_row["model_name"],
    }
)

st.subheader("批量预测结果预览")
st.dataframe(predictions.head(100), use_container_width=True)

st.download_button(
    label="下载预测结果 CSV",
    data=predictions.to_csv(index=False).encode("utf-8-sig"),
    file_name="purchase_predictions.csv",
    mime="text/csv",
)

st.info(
    "建议先执行 `python src/data_preprocessing.py`、`python src/feature_engineering.py` 和 "
    "`python src/train.py`，再启动当前 Streamlit 页面。"
)
