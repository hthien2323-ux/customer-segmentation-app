import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Hệ Thống Phân Tích Doanh Thu & Khách Hàng",
    page_icon=None,
    layout="wide",
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
        font-size: 28px;
        font-weight: 800;
        color: #00e5ff;
        text-align: center;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #00e5ff !important;
        font-size: 18px !important;
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
        font-size: 14px;
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
    '<p class="main-title">HỆ THỐNG PHÂN TÍCH DOANH THU & KHÁCH HÀNG</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Bảng điều khiển quản trị chuẩn hóa dữ liệu Anh (£) - Đa thuật toán</p>',
    unsafe_allow_html=True,
)

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.markdown(
    "<h3 style='font-size:16px !important; border:none;'>Cấu hình hệ thống</h3>",
    unsafe_allow_html=True,
)
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file Online_Retail.csv", type=["csv"]
)

st.sidebar.markdown(
    "<h3 style='font-size:15px !important; border:none; margin-top:15px"
    " !important;'>Bộ lọc kinh doanh</h3>",
    unsafe_allow_html=True,
)
n_clusters_input = st.sidebar.slider("Số lượng nhóm khách hàng (K-Means/Hierarchical)", 2, 8, 3)

# Giai đoạn 4: Bộ chọn thuật toán để đồng bộ toàn hệ thống
selected_algorithm = st.sidebar.selectbox(
    "Chọn thuật toán phân tích",
    ["K-Means", "Hierarchical Clustering", "DBSCAN"],
)

if uploaded_file is not None:
  @st.cache_data
  def load_data(file):
    data = pd.read_csv(file, encoding="ISO-8859-1")
    data.columns = data.columns.str.strip()
    return data

  try:
    df_raw = load_data(uploaded_file)
  except Exception as e:
    st.error(f"Lỗi khi đọc file đầu vào: {e}")
    st.stop()

  initial_rows = len(df_raw)

  # --- GIAI ĐOẠN 1 & 4: TỰ ĐỘNG CHUẨN HÓA & KIỂM TRA LỖI FILE ---
  rename_dict = {}
  for col in df_raw.columns:
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

  df = df_raw.rename(columns=rename_dict)
  required_cols = ["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"]
  missing = [c for c in required_cols if c not in df.columns]
  if missing:
    st.error(f"File đầu vào không hợp lệ! Thiếu các cột bắt buộc: {missing}. Vui lòng kiểm tra lại cấu trúc file.")
    st.stop()

  # --- XỬ LÝ & KIỂM SOÁT CHẤT LƯỢNG DỮ LIỆU (Giai đoạn 1 & 2) ---
  df["InvoiceNo_Str"] = df["InvoiceNo"].astype(str)
  
  # Xác định hóa đơn hủy (bắt đầu bằng 'C') và giao dịch có số lượng <= 0
  cancelled_mask = df["InvoiceNo_Str"].str.startswith("C") | (df["Quantity"] <= 0)
  cancelled_count = cancelled_mask.sum()

  # Loại bỏ dòng thiếu CustomerID, hóa đơn hủy và giá/số lượng không hợp lệ
  df_clean = df.dropna(subset=["CustomerID"])
  df_valid = df_clean[
      ~df_clean["InvoiceNo_Str"].str.startswith("C") & 
      (df_clean["Quantity"] > 0) & 
      (df_clean["UnitPrice"] > 0)
  ].copy()
  
  valid_rows = len(df_valid)
  
  df_valid["InvoiceDate"] = pd.to_datetime(df_valid["InvoiceDate"], errors='coerce')
  df_valid = df_valid.dropna(subset=["InvoiceDate"])
  
  date_min = df_valid["InvoiceDate"].min().strftime("%Y-%m-%d")
  date_max = df_valid["InvoiceDate"].max().strftime("%Y-%m-%d")

  # Công thức tính doanh thu: TotalSum = Quantity * UnitPrice
  df_valid["TotalSum"] = df_valid["Quantity"] * df_valid["UnitPrice"]
  
  # Ngày tham chiếu (Snapshot date) để tính Recency = Ngày max + 1 ngày
  snapshot_date = df_valid["InvoiceDate"].max() + pd.Timedelta(days=1)

  # Tạo bảng RFM chuẩn hóa theo từng khách hàng duy nhất
  rfm = (
      df_valid.groupby("CustomerID")
      .agg({
          "InvoiceDate": lambda x: (snapshot_date - x.max()).days, # Recency: số ngày mua gần nhất
          "InvoiceNo": "nunique",                                    # Frequency: số hóa đơn duy nhất
          "TotalSum": "sum",                                         # Monetary: tổng chi tiêu
      })
      .reset_index()
  )
  rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
  rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]
  valid_customers = len(rfm)

  # --- GIAI ĐOẠN 2: CHẠY 3 THUẬT TOÁN PHÂN CỤM ---
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

  rfm["Cluster_KMeans"] = KMeans(
      n_clusters=n_clusters_input, random_state=42, n_init=10
  ).fit_predict(rfm_scaled)
  
  rfm["Cluster_Hierarchical"] = AgglomerativeClustering(
      n_clusters=n_clusters_input, linkage='ward'
  ).fit_predict(rfm_scaled)
  
  rfm["Cluster_DBSCAN"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(
      rfm_scaled
  )

  # Ánh xạ tên thuật toán sang cột phân cụm tương ứng
  cluster_col_map = {
      "K-Means": "Cluster_KMeans",
      "Hierarchical Clustering": "Cluster_Hierarchical",
      "DBSCAN": "Cluster_DBSCAN",
  }
  active_cluster_col = cluster_col_map[selected_algorithm]

  # --- HIỂN THỊ THỐNG KÊ TIỀN XỬ LÝ (Giai đoạn 4) ---
  with st.expander("📊 Chi tiết tiền xử lý & Kiểm soát chất lượng dữ liệu (Data Quality Summary)"):
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
      st.markdown(f"**Số dòng dữ liệu ban đầu:** {initial_rows:,}")
      st.markdown(f"**Số dòng dữ liệu hợp lệ:** {valid_rows:,}")
    with col_q2:
      st.markdown(f"**Số hóa đơn hủy / loại bỏ:** {cancelled_count:,}")
      st.markdown(f"**Số khách hàng duy nhất hợp lệ:** {valid_customers:,}")
    with col_q3:
      st.markdown(f"**Khoảng thời gian phân tích:** {date_min} đến {date_max}")
      st.markdown("**Quy tắc tính:** $\\text{Revenue} = \\sum (Quantity \\times UnitPrice)$ (Đơn vị: Bảng Anh £)")

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CHỈ SỐ TỔNG QUAN (Chuẩn hóa đơn vị tiền tệ GBP £ - Giai đoạn 1) ---
  total_revenue = rfm["Monetary"].sum()
  total_customers = len(rfm)

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Tổng doanh thu (£)", f"£{total_revenue:,.2f}")
  with col2:
    st.metric("Tổng khách hàng", f"{total_customers:,}")
  with col3:
    st.metric("Doanh thu TB / Khách (£)", f"£{total_revenue/total_customers:,.2f}")
  with col4:
    st.metric("Thuật toán đang chọn", selected_algorithm)

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CÁC TAB QUẢN TRỊ (Giai đoạn 3 & 4) ---
  tab1, tab2, tab3 = st.tabs([
      "Tổng quan Doanh thu",
      f"Hiệu suất Từng Nhóm ({selected_algorithm})",
      "Danh sách Khách hàng",
  ])

  with tab1:
    st.markdown("<h3>GÓC NHÌN ĐA CHIỀU HÀNH VI KHÁCH HÀNG</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Biểu đồ phân tán trực quan hóa hành vi Recency (Trục ngang) và Monetary (Trục dọc) qua 3 phương pháp:</p>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
      st.markdown(
          "<p style='color: #fbbf24; font-weight: 700; font-size: 14px; margin: 0 0 5px 0;'>1. Phân nhóm khách mua (K-Means)</p>",
          unsafe_allow_html=True,
      )
      fig_km = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_KMeans"].astype(str),
          template="plotly_dark",
          labels={"color": "Nhóm", "Recency": "Số ngày mua gần nhất", "Monetary": "Tổng chi tiêu (£)"}
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
          "<p style='color: #fbbf24; font-weight: 700; font-size: 14px; margin: 0 0 5px 0;'>2. Phân loại theo cấp (Hierarchical)</p>",
          unsafe_allow_html=True,
      )
      fig_hi = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_Hierarchical"].astype(str),
          template="plotly_dark",
          labels={"color": "Nhóm", "Recency": "Số ngày mua gần nhất", "Monetary": "Tổng chi tiêu (£)"}
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
          "<p style='color: #fbbf24; font-weight: 700; font-size: 14px; margin: 0 0 5px 0;'>3. Lọc khách hàng VIP (DBSCAN)</p>",
          unsafe_allow_html=True,
      )
      fig_db = px.scatter(
          rfm,
          x="Recency",
          y="Monetary",
          color=rfm["Cluster_DBSCAN"].astype(str),
          template="plotly_dark",
          labels={"color": "Nhóm", "Recency": "Số ngày mua gần nhất", "Monetary": "Tổng chi tiêu (£)"}
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
    st.markdown(f"<h3>BÁO CÁO ĐÓNG GÓP DOANH THU THEO {selected_algorithm.upper()}</h3>", unsafe_allow_html=True)
    
    # Giai đoạn 3: Bổ sung trung vị (median) bên cạnh giá trị trung bình (mean) và đồng bộ theo thuật toán chọn
    revenue_summary = (
        rfm.groupby(active_cluster_col)
        .agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency", "mean"),
            Median_Recency=("Recency", "median"),
            Avg_Frequency=("Frequency", "mean"),
            Median_Frequency=("Frequency", "median"),
            Avg_Monetary=("Monetary", "mean"),
            Median_Monetary=("Monetary", "median"),
        )
        .reset_index()
    )
    
    revenue_summary["Revenue_Share(%)"] = (
        revenue_summary["Total_Revenue"] / total_revenue
    ) * 100

    revenue_summary = revenue_summary.rename(
        columns={
            active_cluster_col: "Nhóm Khách Hàng",
            "Customer_Count": "Số lượng KH (Người)",
            "Total_Revenue": "Tổng doanh thu (£)",
            "Revenue_Share(%)": "Tỷ trọng đóng góp (%)",
            "Avg_Recency": "Số ngày mua gần nhất (TB)",
            "Median_Recency": "Số ngày mua gần nhất (Trung vị)",
            "Avg_Frequency": "Tần suất mua (TB)",
            "Median_Frequency": "Tần suất mua (Trung vị)",
            "Avg_Monetary": "Chi tiêu KH (TB £)",
            "Median_Monetary": "Chi tiêu KH (Trung vị £)",
        }
    )

    st.dataframe(
        revenue_summary.style.format({
            "Tổng doanh thu (£)": "{:,.2f}",
            "Tỷ trọng đóng góp (%)": "{:.2f}%",
            "Số ngày mua gần nhất (TB)": "{:.1f}",
            "Số ngày mua gần nhất (Trung vị)": "{:.1f}",
            "Tần suất mua (TB)": "{:.1f}",
            "Tần suất mua (Trung vị)": "{:.1f}",
            "Chi tiêu KH (TB £)": "{:,.2f}",
            "Chi tiêu KH (Trung vị £)": "{:,.2f}",
            "Số lượng KH (Người)": "{:,}",
        }),
        use_container_width=True,
    )

    fig_rev = px.bar(
        revenue_summary,
        x="Nhóm Khách Hàng",
        y="Tổng doanh thu (£)",
        text="Tỷ trọng đóng góp (%)",
        title=f"Biểu đồ phân bổ doanh thu theo nhóm ({selected_algorithm})",
        template="plotly_dark",
        labels={"Tổng doanh thu (£)": "Tổng doanh thu (£)", "Nhóm Khách Hàng": "Nhóm Khách Hàng"}
    )
    fig_rev.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_rev.update_layout(
        plot_bgcolor="#0b0f19",
        paper_bgcolor="#0b0f19",
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Giai đoạn 3: Bổ sung gợi ý hành động kinh doanh tương ứng từng phân khúc (chưa kiểm chứng thực nghiệm)
    st.markdown("<h3>GỢI Ý HÀNH ĐỘNG CHIẾN LƯỢC CHO TỪNG NHÓM (CHƯA KIỂM CHỨNG THỰC NGHIỆM)</h3>", unsafe_allow_html=True)
    st.info("Lưu ý: Các đề xuất dưới đây dựa trên phân tích đặc tính RFM thực tế và cần được thử nghiệm A/B testing trước khi triển khai quy mô lớn.")
    
    for idx, row in revenue_summary.iterrows():
      group_id = row["Nhóm Khách Hàng"]
      share = row["Tỷ trọng đóng góp (%)"]
      count = row["Số lượng KH (Người)"]
      st.markdown(f"* **Nhóm {group_id}** (Quy mô: {count:,} khách hàng, Đóng góp: {share:.2f}% doanh thu): Tập trung duy trì tương tác thường xuyên, chăm sóc đặc quyền nếu là nhóm giá trị cao hoặc kích hoạt lại nếu số ngày mua gần nhất cao.")

  with tab3:
    st.markdown("<h3>CHI TIẾT KHÁCH HÀNG THEO DỮ LIỆU</h3>", unsafe_allow_html=True)
    
    # Giai đoạn 4: Thêm tính năng tìm kiếm và xuất file CSV danh sách khách hàng
    search_query = st.text_input("🔍 Tìm kiếm theo Mã Khách Hàng (CustomerID):", "")
    filtered_rfm = rfm.copy()
    if search_query:
      filtered_rfm = filtered_rfm[filtered_rfm["CustomerID"].astype(str).str.contains(search_query, na=False)]

    st.dataframe(filtered_rfm, use_container_width=True)

    csv_data = filtered_rfm.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải xuống danh sách khách hàng (.csv)",
        data=csv_data,
        file_name="customer_segmentation_results.csv",
        mime="text/csv",
    )

else:
  st.markdown(
      """
      <div style="text-align: center; padding: 50px; background-color: #111827; border-radius: 8px; border: 1px dashed #374151; margin-top: 40px;">
          <h3 style="color: #00e5ff; border: none; margin-bottom: 10px;">CHƯA CÓ DỮ LIỆU ĐƯỢC TẢI LÊN</h3>
          <p style="font-size: 15px; color: #9ca3af;">Vui lòng tải tệp <b>Online_Retail.csv</b> ở thanh bên trái để khởi chạy hệ thống phân tích.</p>
      </div>
      """,
      unsafe_allow_html=True,
  )