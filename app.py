import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Hệ Thống Phân Khúc & Phân Tích Doanh Thu Khách Hàng",
    page_icon=None,
    layout="wide",
)

# --- TÙY CHỈNH CSS (ĐẲNG CẤP DOANH NGHIỆP) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #00e5ff;
        text-align: center;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 15px;
        font-weight: 600;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #00e5ff !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 6px;
        margin-top: 15px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: #0b0f19;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 15px;
        font-weight: 700;
        padding: 8px 18px;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e5ff !important;
        color: #0b0f19 !important;
        border: 1px solid #00e5ff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    /* Màu Hồng Neon nổi bật cho Tổng doanh thu & Khách hàng */
    [data-testid="stMetric"]:nth-of-type(1) [data-testid="stMetricLabel"],
    [data-testid="stMetric"]:nth-of-type(2) [data-testid="stMetricLabel"] {
        color: #ff007f !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"]:nth-of-type(1) [data-testid="stMetricValue"],
    [data-testid="stMetric"]:nth-of-type(2) [data-testid="stMetricValue"] {
        color: #ff007f !important;
        text-shadow: 0 0 10px rgba(255, 0, 127, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- TIÊU ĐỀ TRANG ---
st.markdown(
    '<p class="main-title">HỆ THỐNG PHÂN TÍCH DOANH THU & PHÂN KHÚC KHÁCH HÀNG</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Bảng điều khiển quản trị chiến lược tích hợp thuật toán đa chiều (K-Means, Hierarchical, DBSCAN)</p>',
    unsafe_allow_html=True,
)

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.markdown(
    "<h3 style='font-size:18px !important; border:none;'>Cấu hình hệ thống</h3>",
    unsafe_allow_html=True,
)
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file Online_Retail.csv", type=["csv"]
)

st.sidebar.markdown(
    "<h3 style='font-size:16px !important; border:none; margin-top:15px"
    " !important;'>Bộ lọc chiến lược</h3>",
    unsafe_allow_html=True,
)
n_clusters_input = st.sidebar.slider("Số lượng phân khúc khách hàng", 2, 8, 3)

if uploaded_file is not None:
  @st.cache_data
  def load_data(file):
    data = pd.read_csv(file, encoding="ISO-8859-1")
    data.columns = data.columns.str.strip()
    return data

  df = load_data(uploaded_file)

  # --- TỰ ĐỘNG CHUẨN HÓA TÊN CỘT ---
  rename_dict = {}
  for col in df.columns:
    c_lower = col.lower().replace(" ", "").replace("_", "")
    if "customer" in c_lower:
      rename_dict[col] = "CustomerID"
    elif "invoicedate" in c_lower or "date" in c_lower:
      rename_dict[col] = "InvoiceDate"
    elif "invoiceno" in c_lower or "invoice" in c_lower:
      rename_dict[col] = "InvoiceNo"
    elif "quantity" in c_lower:
      rename_dict[col] = "Quantity"
    elif "unitprice" in c_lower or "price" in c_lower:
      rename_dict[col] = "UnitPrice"

  df = df.rename(columns=rename_dict)
  required_cols = ["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"]
  missing = [c for c in required_cols if c not in df.columns]
  if missing:
    st.error(f"File thiếu cột bắt buộc: {missing}")
    st.stop()

  # --- XỬ LÝ DỮ LIỆU & RFM ---
  df = df.dropna(subset=["CustomerID"])
  df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
  df["TotalSum"] = df["Quantity"] * df["UnitPrice"]
  snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

  rfm = (
      df.groupby("CustomerID")
      .agg({
          "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
          "InvoiceNo": "nunique",
          "TotalSum": "sum",
      })
      .reset_index()
  )
  rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
  rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]

  # --- CHẠY 3 THUẬT TOÁN ĐỂ TỔNG HỢP ---
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

  rfm["Cluster_KMeans"] = KMeans(
      n_clusters=n_clusters_input, random_state=42, n_init=10
  ).fit_predict(rfm_scaled)
  rfm["Cluster_Hierarchical"] = AgglomerativeClustering(
      n_clusters=n_clusters_input
  ).fit_predict(rfm_scaled)
  rfm["Cluster_DBSCAN"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(
      rfm_scaled
  )

  # --- CÁC CHỈ SỐ TỔNG QUAN DOANH THU (EXECUTIVE METRICS) ---
  total_revenue = rfm["Monetary"].sum()
  total_customers = len(rfm)
  avg_order_value = total_revenue / rfm["Frequency"].sum()

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Tổng doanh thu hệ thống", f"${total_revenue:,.0f}")
  with col2:
    st.metric("Tổng khách hàng phân tích", f"{total_customers:,}")
  with col3:
    st.metric("Giá trị vòng đời TB / KH", f"${total_revenue/total_customers:,.2f}")
  with col4:
    st.metric("Mô hình hợp nhất", "3 Thuật toán (AI Core)")

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CÁC TAB QUẢN TRỊ ---
  tab1, tab2, tab3 = st.tabs([
      "Tổng quan Doanh thu & Phân khúc",
      "Hiệu suất & Đóng góp Doanh thu từng Nhóm",
      "Dữ liệu khách hàng chi tiết",
  ])

  with tab1:
    st.markdown("<h3>BIỂU ĐỒ PHÂN TÍCH ĐA MÔ HÌNH HÀNH VI KHÁCH HÀNG</h3>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
      st.markdown(
          "<p"
          " style='color: #fbbf24; font-weight: 700; font-size: 15px; margin:"
          " 0 0 5px 0;'>1. K-Means (Tối ưu biên độ giá trị)</p>",
          unsafe_allow_html=True,
      )
      fig_km = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_KMeans"].astype(str),
          template="plotly_dark",
      )
      fig_km.update_layout(
          plot_bgcolor="#0b0f19",
          paper_bgcolor="#0b0f19",
          font=dict(size=10),
          showlegend=False,
          margin=dict(l=10, r=10, t=10, b=10),
      )
      st.plotly_chart(fig_km, use_container_width=True)

    with col_b:
      st.markdown(
          "<p"
          " style='color: #fbbf24; font-weight: 700; font-size: 15px; margin:"
          " 0 0 5px 0;'>2. Hierarchical (Cấu trúc phân tầng)</p>",
          unsafe_allow_html=True,
      )
      fig_hi = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_Hierarchical"].astype(str),
          template="plotly_dark",
      )
      fig_hi.update_layout(
          plot_bgcolor="#0b0f19",
          paper_bgcolor="#0b0f19",
          font=dict(size=10),
          showlegend=False,
          margin=dict(l=10, r=10, t=10, b=10),
      )
      st.plotly_chart(fig_hi, use_container_width=True)

    with col_c:
      st.markdown(
          "<p"
          " style='color: #fbbf24; font-weight: 700; font-size: 15px; margin:"
          " 0 0 5px 0;'>3. DBSCAN (Cô lập khách hàng VIP dị biệt)</p>",
          unsafe_allow_html=True,
      )
      fig_db = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_DBSCAN"].astype(str),
          template="plotly_dark",
      )
      fig_db.update_layout(
          plot_bgcolor="#0b0f19",
          paper_bgcolor="#0b0f19",
          font=dict(size=10),
          showlegend=False,
          margin=dict(l=10, r=10, t=10, b=10),
      )
      st.plotly_chart(fig_db, use_container_width=True)

  with tab2:
    st.markdown("<h3>BÁO CÁO ĐÓNG GÓP DOANH THU & CHIẾN LƯỢC THEO NHÓM</h3>", unsafe_allow_html=True)
    
    # Tổng hợp số liệu theo K-Means làm chuẩn tài chính
    revenue_summary = (
        rfm.groupby("Cluster_KMeans")
        .agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .reset_index()
    )
    
    revenue_summary["Revenue_Share(%)"] = (
        revenue_summary["Total_Revenue"] / total_revenue
    ) * 100

    revenue_summary = revenue_summary.rename(
        columns={
            "Cluster_KMeans": "Nhóm Phân Khúc",
            "Customer_Count": "Số lượng KH",
            "Total_Revenue": "Tổng doanh thu ($)",
            "Revenue_Share(%)": "Tỷ trọng doanh thu (%)",
            "Avg_Recency": "Số ngày mua gần nhất (TB)",
            "Avg_Frequency": "Tần suất mua (TB)",
            "Avg_Monetary": "Chi tiêu TB / KH ($)",
        }
    )

    st.dataframe(
        revenue_summary.style.format({
            "Tổng doanh thu ($)": "{:,.2f}",
            "Tỷ trọng doanh thu (%)": "{:.2f}%",
            "Số ngày mua gần nhất (TB)": "{:.1f}",
            "Tần suất mua (TB)": "{:.1f}",
            "Chi tiêu TB / KH ($)": "{:,.2f}",
            "Số lượng KH": "{:,}",
        }),
        use_container_width=True,
    )

    # Biểu đồ trực quan tỷ trọng doanh thu đóng góp
    fig_rev = px.bar(
        revenue_summary,
        x="Nhóm Phân Khúc",
        y="Tổng doanh thu ($)",
        text="Tỷ trọng doanh thu (%)",
        title="Biểu đồ phân bổ nguồn lực doanh thu theo phân khúc khách hàng",
        template="plotly_dark",
    )
    fig_rev.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_rev.update_layout(
        plot_bgcolor="#0b0f19",
        paper_bgcolor="#0b0f19",
    )
    st.plotly_chart(fig_rev, use_container_width=True)

  with tab3:
    st.markdown("<h3>CHI TIẾT DỮ LIỆU TÀI CHÍNH & KHÁCH HÀNG ĐÃ PHÂN KHÚC</h3>", unsafe_allow_html=True)
    st.dataframe(rfm, use_container_width=True)

else:
  st.markdown(
      """
      <div style="text-align: center; padding: 50px; background-color: #111827; border-radius: 8px; border: 1px dashed #374151; margin-top: 40px;">
          <h3 style="color: #00e5ff; border: none; margin-bottom: 10px;">CHƯA CÓ DỮ LIỆU KINH DOANH ĐƯỢC TẢI LÊN</h3>
          <p style="font-size: 15px; color: #9ca3af;">Vui lòng tải tệp dữ liệu <b>Online_Retail.csv</b> ở thanh điều hướng bên trái để kích hoạt hệ thống phân tích doanh thu.</p>
      </div>
      """,
      unsafe_allow_html=True,
  )