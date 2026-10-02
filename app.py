import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Hệ Thống Phân Khúc KH - Đa Thuật Toán", page_icon=None, layout="wide"
)

# --- TÙY CHỈNH CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #00e5ff;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 15px;
        font-weight: 600;
        color: #93c5fd;
        text-align: center;
        margin-bottom: 25px;
    }
    h3 {
        color: #00e5ff !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 6px;
        margin-top: 20px !important;
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
    .note-box {
        background-color: #111827;
        border-left: 4px solid #00e5ff;
        padding: 12px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 14px;
        color: #e5e7eb;
        line-height: 1.5;
    }
    .algo-desc {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 10px;
        border-radius: 6px;
        font-size: 13px;
        color: #cbd5e1;
        margin-bottom: 10px;
        min-height: 75px;
    }
    .note-title {
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
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
    /* Hồng Neon cho Tổng khách hàng */
    [data-testid="stMetric"]:nth-of-type(1) [data-testid="stMetricLabel"] {
        color: #ff007f !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"]:nth-of-type(1) [data-testid="stMetricValue"] {
        color: #ff007f !important;
        text-shadow: 0 0 10px rgba(255, 0, 127, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- TIÊU ĐỀ TRANG ---
st.markdown(
    '<p class="main-title">HỆ THỐNG PHÂN KHÚC KHÁCH HÀNG TÍCH HỢP 3 THUẬT TOÁN</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Kết hợp đồng thời K-Means, Hierarchical và DBSCAN nhằm tối ưu hóa nhận diện hành vi khách hàng qua mô hình RFM</p>',
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
    " !important;'>Tham số phân cụm</h3>",
    unsafe_allow_html=True,
)
n_clusters_input = st.sidebar.slider(
    "Số lượng cụm mục tiêu (K-Means & Hierarchical)", 2, 8, 3
)

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

  # --- XỬ LÝ RFM ---
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

  # --- CHẠY 3 THUẬT TOÁN ---
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

  # 1. K-Means
  kmeans_model = KMeans(
      n_clusters=n_clusters_input, random_state=42, n_init=10
  )
  rfm["Cluster_KMeans"] = kmeans_model.fit_predict(rfm_scaled)

  # 2. Hierarchical
  hier_model = AgglomerativeClustering(n_clusters=n_clusters_input)
  rfm["Cluster_Hierarchical"] = hier_model.fit_predict(rfm_scaled)

  # 3. DBSCAN
  dbscan_model = DBSCAN(eps=0.5, min_samples=5)
  rfm["Cluster_DBSCAN"] = dbscan_model.fit_predict(rfm_scaled)

  # --- CÁC CHỈ SỐ TỔNG QUAN ---
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Tổng khách hàng", f"{len(rfm):,}")
  with col2:
    st.metric("Tổng doanh thu", f"${rfm['Monetary'].sum():,.0f}")
  with col3:
    st.metric("Tần suất mua TB", f"{rfm['Frequency'].mean():.1f} lần")
  with col4:
    st.metric("Kiến trúc hệ thống", "Đa mô hình (3 trong 1)")

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CÁC TAB NỘI DUNG ---
  tab1, tab2, tab3 = st.tabs([
      "Trực quan hóa Đa Thuật Toán",
      "Thống kê chi tiết các cụm",
      "Dữ liệu khách hàng tổng hợp",
  ])

  with tab1:
    st.markdown("<h3>GÓC NHÌN ĐẶC TRƯNG CỦA 3 THUẬT TOÁN</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="note-box">
            <div class="note-title">Triết lý thiết kế hệ thống kết hợp</div>
            Hệ thống không phụ thuộc vào duy nhất một mô hình mà kết hợp đồng thời 3 cơ chế học máy khác nhau để bổ khuyết điểm yếu cho nhau: <b>Tối ưu tâm cụm (K-Means)</b>, <b>Phân cấp cấu trúc (Hierarchical)</b> và <b>Phát hiện mật độ & Nhiễu (DBSCAN)</b>. Bạn có thể kéo thanh trượt ở thanh bên trái để thay đổi số lượng cụm và quan sát phản ứng của các mô hình theo thời gian thực!
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
      st.markdown(
          "<p"
          " style='color: #fbbf24; font-weight: 700; font-size: 16px; margin:"
          " 0;'>1. K-Means Clustering</p>",
          unsafe_allow_html=True,
      )
      st.markdown(
          """
            <div class="algo-desc">
                <b>Đặc trưng:</b> Dựa trên tâm cụm (Centroids). Tốc độ cực nhanh, phân chia ranh giới rõ ràng dựa vào khoảng cách tối ưu.
            </div>
            """,
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
          " style='color: #fbbf24; font-weight: 700; font-size: 16px; margin:"
          " 0;'>2. Hierarchical Clustering</p>",
          unsafe_allow_html=True,
      )
      st.markdown(
          """
            <div class="algo-desc">
                <b>Đặc trưng:</b> Phân cấp cấu trúc (Bottom-up). Xây dựng phả hệ nhóm khách hàng dựa trên độ tương đồng gần gũi.
            </div>
            """,
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
          " style='color: #fbbf24; font-weight: 700; font-size: 16px; margin:"
          " 0;'>3. DBSCAN Clustering</p>",
          unsafe_allow_html=True,
      )
      st.markdown(
          """
            <div class="algo-desc">
                <b>Đặc trưng:</b> Dựa trên mật độ không gian. Tự động nhận diện và cô lập các điểm nhiễu (khách hàng VIP bất thường).
            </div>
            """,
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
    st.markdown("<h3>THỐNG KÊ TRUNG BÌNH THEO CỤM (K-MEANS)</h3>", unsafe_allow_html=True)
    cluster_summary = (
        rfm.groupby("Cluster_KMeans")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )
    cluster_summary["Customer_Count"] = (
        rfm.groupby("Cluster_KMeans")["CustomerID"].count().values
    )
    cluster_summary = cluster_summary.rename(
        columns={
            "Cluster_KMeans": "Nhóm Cụm",
            "Recency": "Thời gian mua gần nhất TB (Ngày)",
            "Frequency": "Tần suất mua TB (Lần)",
            "Monetary": "Tổng chi tiêu TB ($)",
            "Customer_Count": "Quy mô khách hàng",
        }
    )
    st.dataframe(
        cluster_summary.style.format({
            "Thời gian mua gần nhất TB (Ngày)": "{:.1f}",
            "Tần suất mua TB (Lần)": "{:.1f}",
            "Tổng chi tiêu TB ($)": "{:,.2f}",
            "Quy mô khách hàng": "{:,}",
        }),
        use_container_width=True,
    )

  with tab3:
    st.markdown("<h3>DANH SÁCH CHI TIẾT DỮ LIỆU & KẾT QUẢ ĐA MÔ HÌNH</h3>", unsafe_allow_html=True)
    st.dataframe(rfm, use_container_width=True)

else:
  st.markdown(
      """
      <div style="text-align: center; padding: 40px; background-color: #111827; border-radius: 8px; border: 1px dashed #374151; margin-top: 40px;">
          <h3 style="color: #00e5ff; border: none; margin-bottom: 10px;">CHƯA CÓ DỮ LIỆU TẢI LÊN</h3>
          <p style="font-size: 15px; color: #9ca3af;">Hãy tải file <b>Online_Retail.csv</b> lên ở thanh menu bên trái để khởi chạy hệ thống tích hợp đa thuật toán.</p>
      </div>
      """,
      unsafe_allow_html=True,
  )