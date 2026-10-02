import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Phân Khúc Khách Hàng TMĐT", page_icon=None, layout="wide"
)

# --- TÙY CHỈNH CSS (DARK EDITION, CHỮ TO, KHÔNG ICON, MÀU SẮC NỔI BẬT, DỄ NHÌN) ---
st.markdown(
    """
    <style>
    /* Tổng thể nền và font chữ */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Tiêu đề chính */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #00e5ff;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        font-weight: 600;
        color: #93c5fd;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Tiêu đề các mục lớn */
    h3 {
        color: #00e5ff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 8px;
        margin-top: 25px !important;
    }
    
    /* Màu sắc các tab hiển thị rõ ràng, không bị chìm */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: #0b0f19;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 700;
        padding: 10px 20px;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e5ff !important;
        color: #0b0f19 !important;
        border: 1px solid #00e5ff !important;
    }

    /* Các khối chú thích và giải thích trực quan */
    .note-box {
        background-color: #111827;
        border-left: 4px solid #00e5ff;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
        font-size: 16px;
        color: #e5e7eb;
        line-height: 1.6;
    }
    .note-title {
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 5px;
        font-size: 17px;
    }

    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- TIÊU ĐỀ TRANG ---
st.markdown(
    '<p class="main-title">PHÂN KHÚC KHÁCH HÀNG THƯƠNG MẠI ĐIỆN TỬ</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Hệ thống phân tích hành vi khách hàng nâng cao kết hợp Học máy và Chỉ số RFM</p>',
    unsafe_allow_html=True,
)

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.markdown(
    "<h3 style='font-size:20px !important; border:none;'>Cấu hình hệ thống</h3>",
    unsafe_allow_html=True,
)
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file Online_Retail.csv", type=["csv"]
)

if uploaded_file is not None:
  @st.cache_data
  def load_data(file):
    return pd.read_csv(file, encoding="ISO-8859-1")

  df = load_data(uploaded_file)

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

  # --- SIDEBAR THUẬT TOÁN ---
  st.sidebar.markdown(
      "<h3 style='font-size:20px !important; border:none; margin-top:20px"
      " !important;'>Thuật toán Phân cụm</h3>",
      unsafe_allow_html=True,
  )
  algorithm = st.sidebar.selectbox(
      "Chọn thuật toán", ["K-Means", "Hierarchical", "DBSCAN"]
  )

  if algorithm == "K-Means":
    n_clusters = st.sidebar.slider("Số lượng cụm", 2, 8, 4)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)
  elif algorithm == "Hierarchical":
    n_clusters = st.sidebar.slider("Số lượng cụm", 2, 8, 4)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    model = AgglomerativeClustering(n_clusters=n_clusters)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)
  else:
    eps = st.sidebar.slider("EPS", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    model = DBSCAN(eps=eps, min_samples=min_samples)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)

  # --- CÁC CHỈ SỐ TỔNG QUAN (METRICS) ---
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Tổng khách hàng", f"{len(rfm):,}")
  with col2:
    st.metric("Tổng doanh thu", f"${rfm['Monetary'].sum():,.0f}")
  with col3:
    st.metric("Tần suất mua TB", f"{rfm['Frequency'].mean():.1f} lần")
  with col4:
    st.metric("Số cụm phân tách", f"{rfm['Cluster'].nunique()}")

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CÁC TAB NỘI DUNG (MÀU SẮC RÕ RÀNG, CHỮ TO) ---
  tab1, tab2, tab3 = st.tabs(
      ["Trực quan hóa và Phân tích", "Thống kê chi tiết cụm", "Dữ liệu khách hàng"]
  )

  with tab1:
    st.markdown("<h3>BIỂU ĐỒ TƯƠNG TÁC VÀ TỶ LỆ PHÂN KHÚC</h3>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="note-box">
            <div class="note-title">Hướng dẫn đọc biểu đồ</div>
            Biểu đồ phân tán thể hiện mối quan hệ giữa thời gian mua hàng gần nhất (Recency) và tổng chi tiêu (Monetary). 
            Biểu đồ tròn bên cạnh minh họa tỷ trọng phân bổ số lượng khách hàng của từng nhóm phân khúc trên tổng thể hệ thống.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([2, 1])

    with col_l:
      fig_scatter = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster"].astype(str),
          title="Mối quan hệ giữa Số ngày mua gần đây và Tổng chi tiêu",
          template="plotly_dark",
          color_discrete_sequence=px.colors.qualitative.Set2,
      )
      fig_scatter.update_layout(
          plot_bgcolor="#0b0f19",
          paper_bgcolor="#0b0f19",
          font=dict(color="#f3f4f6", size=13),
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

    with col_r:
      fig_pie = px.pie(
          rfm,
          names=rfm["Cluster"].astype(str),
          title="Tỷ lệ phân bổ các cụm",
          template="plotly_dark",
          color_discrete_sequence=px.colors.qualitative.Set2,
      )
      fig_pie.update_layout(
          plot_bgcolor="#0b0f19",
          paper_bgcolor="#0b0f19",
          font=dict(color="#f3f4f6", size=13),
      )
      st.plotly_chart(fig_pie, use_container_width=True)

  with tab2:
    st.markdown("<h3>BẢNG TỔNG HỢP CHỈ SỐ TRUNG BÌNH THEO PHÂN KHÚC</h3>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="note-box">
            <div class="note-title">Giải thích bảng số liệu</div>
            Bảng dưới đây tổng hợp các giá trị trung bình về chỉ số RFM và quy mô số lượng khách hàng của từng nhóm cụm, 
            giúp doanh nghiệp đánh giá chính xác giá trị và hành vi của từng phân khúc khách hàng.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cluster_summary = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )
    cluster_summary["Customer_Count"] = (
        rfm.groupby("Cluster")["CustomerID"].count().values
    )

    cluster_summary = cluster_summary.rename(
        columns={
            "Recency": "Recency TB (Ngày)",
            "Frequency": "Frequency TB (Lần)",
            "Monetary": "Monetary TB ($)",
            "Customer_Count": "Số lượng KH",
        }
    )

    st.dataframe(
        cluster_summary.style.format({
            "Recency TB (Ngày)": "{:.1f}",
            "Frequency TB (Lần)": "{:.1f}",
            "Monetary TB ($)": "{:,.2f}",
            "Số lượng KH": "{:,}",
        }),
        use_container_width=True,
    )

  with tab3:
    st.markdown("<h3>DANH SÁCH CHI TIẾT RFM CỦA KHÁCH HÀNG</h3>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="note-box">
            <div class="note-title">Dữ liệu chi tiết</div>
            Danh sách toàn bộ mã khách hàng đi kèm các chỉ số thành phần Recency, Frequency, Monetary và nhãn phân cụm tương ứng.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.dataframe(rfm, use_container_width=True)

else:
  st.markdown(
      """
      <div style="text-align: center; padding: 50px; background-color: #111827; border-radius: 8px; border: 1px dashed #374151; margin-top: 50px;">
          <h3 style="color: #00e5ff; border: none; margin-bottom: 10px;">CHƯA CÓ DỮ LIỆU ĐƯỢC TẢI LÊN</h3>
          <p style="font-size: 16px; color: #9ca3af;">Vui lòng tải lên file dữ liệu <b>Online_Retail.csv</b> ở thanh cấu hình phía bên trái để khởi chạy hệ thống phân tích.</p>
      </div>
      """,
      unsafe_allow_html=True,
  )