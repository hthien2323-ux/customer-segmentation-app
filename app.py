import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
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
    /* Sửa lỗi Tab mờ và khó nhìn */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0b0f19;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px;
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        padding: 8px 16px;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e5ff !important;
        color: #0b0f19 !important;
        font-weight: 700;
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
    '<p class="sub-title">Bảng điều khiển quản trị chuẩn hóa dữ liệu Anh (£) - Đa thuật toán & Gợi ý chiến lược</p>',
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
    "<h3 style='font-size:15px !important; border:none; margin-top:15px !important;'>Bộ lọc kinh doanh</h3>",
    unsafe_allow_html=True,
)

# Bộ chọn thuật toán
selected_algorithm = st.sidebar.selectbox(
    "Chọn thuật toán phân tích",
    ["K-Means", "Hierarchical Clustering", "DBSCAN"],
)

# Ẩn hoặc vô hiệu hóa slider số nhóm khi dùng DBSCAN
if selected_algorithm == "DBSCAN":
  st.sidebar.info("DBSCAN tự động xác định số nhóm dựa trên mật độ. Slider số nhóm tạm vô hiệu hóa.")
  n_clusters_input = 3 # Giá trị mặc định ẩn
  eps_val = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.1)
  min_samples_val = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)
else:
  n_clusters_input = st.sidebar.slider("Số lượng nhóm khách hàng", 2, 8, 3)
  eps_val, min_samples_val = 0.5, 5

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

  # --- TỰ ĐỘNG CHUẨN HÓA CỘT ---
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
    st.error(f"File đầu vào không hợp lệ! Thiếu các cột bắt buộc: {missing}.")
    st.stop()

  # --- XỬ LÝ LỌC ĐƠN HỦY & CHẤT LƯỢNG DỮ LIỆU ---
  df["InvoiceNo_Str"] = df["InvoiceNo"].astype(str)
  
  # Lọc hóa đơn hủy (bắt đầu bằng 'C') và số lượng <= 0 (như trường hợp CustomerID 12346)
  cancelled_mask = df["InvoiceNo_Str"].str.startswith("C") | (df["Quantity"] <= 0)
  cancelled_count = cancelled_mask.sum()

  df_clean = df.dropna(subset=["CustomerID"])
  df_valid = df_clean[
      ~df_clean["InvoiceNo_Str"].str.startswith("C") & 
      (df_clean["Quantity"] > 0) & 
      (df_valid_price := df_clean["UnitPrice"] > 0)
  ].copy()
  
  valid_rows = len(df_valid)
  
  df_valid["InvoiceDate"] = pd.to_datetime(df_valid["InvoiceDate"], errors='coerce')
  df_valid = df_valid.dropna(subset=["InvoiceDate"])
  
  date_min = df_valid["InvoiceDate"].min().strftime("%Y-%m-%d")
  date_max = df_valid["InvoiceDate"].max().strftime("%Y-%m-%d")

  df_valid["TotalSum"] = df_valid["Quantity"] * df_valid["UnitPrice"]
  snapshot_date = df_valid["InvoiceDate"].max() + pd.Timedelta(days=1)

  # Tạo bảng RFM
  rfm = (
      df_valid.groupby("CustomerID")
      .agg({
          "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
          "InvoiceNo": "nunique",
          "TotalSum": "sum",
      })
      .reset_index()
  )
  rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
  rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]
  valid_customers = len(rfm)

  # --- LOG-TRANSFORM & CHUẨN HÓA DỮ LIỆU ---
  # Khắc phục độ lệch phải mạnh của phân phối R, F, M
  rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm_log)

  # Chạy 3 thuật toán
  rfm["Cluster_KMeans"] = KMeans(
      n_clusters=n_clusters_input, random_state=42, n_init=10
  ).fit_predict(rfm_scaled)
  
  rfm["Cluster_Hierarchical"] = AgglomerativeClustering(
      n_clusters=n_clusters_input, linkage='ward'
  ).fit_predict(rfm_scaled)
  
  rfm["Cluster_DBSCAN"] = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit_predict(
      rfm_scaled
  )

  cluster_col_map = {
      "K-Means": "Cluster_KMeans",
      "Hierarchical Clustering": "Cluster_Hierarchical",
      "DBSCAN": "Cluster_DBSCAN",
  }
  active_cluster_col = cluster_col_map[selected_algorithm]

  # --- HÀM ÁNH XẠ NHÃN SANG TÊN KINH DOANH Ý NGHĨA ---
  def map_cluster_names(df_sub, cluster_col):
    # Sắp xếp các cụm theo Monetary trung bình để đặt tên nhất quán
    grouped_m = df_sub.groupby(cluster_col)["Monetary"].mean().reset_index()
    grouped_m = grouped_m.sort_values(by="Monetary", ascending=False)
    
    mapping = {}
    n_unique = len(grouped_m)
    for idx, row in enumerate(grouped_m.iterrows()):
      c_id = row[1][cluster_col]
      if c_id == -1:
        mapping[c_id] = "Điểm nhiễu / Outliers"
      elif idx == 0:
        mapping[c_id] = "VIP & Khách hàng giá trị cao"
      elif idx == 1 and n_unique > 2:
        mapping[c_id] = "Khách hàng trung thành / Thân thiết"
      else:
        mapping[c_id] = "Khách hàng ngủ đông / Nguy cơ rời bỏ"
    return mapping

  name_mapping = map_cluster_names(rfm, active_cluster_col)
  rfm["Segment_Name"] = rfm[active_cluster_col].map(name_mapping)

  # Tính Silhouette Score đánh giá chất lượng cụm
  try:
    if len(rfm[active_cluster_col].unique()) > 1:
      score = silhouette_score(rfm_scaled, rfm[active_cluster_col])
    else:
      score = 0.0
  except:
    score = 0.0

  # --- HIỂN THỊ THỐNG KÊ TIỀN XỬ LÝ ---
  with st.expander("📊 Chi tiết tiền xử lý & Kiểm soát chất lượng dữ liệu (Data Quality Summary)"):
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
      st.markdown(f"**Số dòng dữ liệu ban đầu:** {initial_rows:,}")
      st.markdown(f"**Số dòng dữ liệu hợp lệ:** {valid_rows:,}")
    with col_q2:
      st.markdown(f"**Số hóa đơn hủy / hoàn trả (C...):** {cancelled_count:,}")
      st.markdown(f"**Số khách hàng duy nhất hợp lệ:** {valid_customers:,}")
    with col_q3:
      st.markdown(f"**Khoảng thời gian phân tích:** {date_min} đến {date_max}")
      st.markdown(f"**Chỉ số Silhouette Score ({selected_algorithm}):** `{score:.3f}`")

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CHỈ SỐ TỔNG QUAN ---
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
    st.metric("Thuật toán", selected_algorithm)

  st.markdown("<br>", unsafe_allow_html=True)

  # --- CÁC TAB QUẢN TRỊ ---
  tab1, tab2, tab3 = st.tabs([
      "Tổng quan Doanh thu & Biểu đồ",
      f"Hiệu suất & Gợi ý hành động ({selected_algorithm})",
      "Danh sách Khách hàng",
  ])

  with tab1:
    st.markdown("<h3>GÓC NHÌN ĐA CHIỀU HÀNH VI KHÁCH HÀNG (LOG SCALE)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Trực quan hóa phân phối hành vi qua Log-transform giúp xử lý triệt để các điểm dữ liệu lớn (>250k):</p>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
      st.markdown("<p style='color: #fbbf24; font-weight: 700; font-size: 14px;'>1. K-Means (Log)</p>", unsafe_allow_html=True)
      fig_km = px.scatter(
          rfm, x="Recency", y="Monetary", color=rfm["Cluster_KMeans"].astype(str),
          template="plotly_dark", log_y=True,
          labels={"color": "Nhóm", "Recency": "Số ngày", "Monetary": "Chi tiêu (£)"}
      )
      fig_km.update_layout(plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19", font=dict(size=10), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
      st.plotly_chart(fig_km, use_container_width=True)

    with col_b:
      st.markdown("<p style='color: #fbbf24; font-weight: 700; font-size: 14px;'>2. Hierarchical (Log)</p>", unsafe_allow_html=True)
      fig_hi = px.scatter(
          rfm, x="Recency", y="Monetary", color=rfm["Cluster_Hierarchical"].astype(str),
          template="plotly_dark", log_y=True,
          labels={"color": "Nhóm", "Recency": "Số ngày", "Monetary": "Chi tiêu (£)"}
      )
      fig_hi.update_layout(plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19", font=dict(size=10), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
      st.plotly_chart(fig_hi, use_container_width=True)

    with col_c:
      st.markdown("<p style='color: #fbbf24; font-weight: 700; font-size: 14px;'>3. DBSCAN (Log)</p>", unsafe_allow_html=True)
      fig_db = px.scatter(
          rfm, x="Recency", y="Monetary", color=rfm["Cluster_DBSCAN"].astype(str),
          template="plotly_dark", log_y=True,
          labels={"color": "Nhóm", "Recency": "Số ngày", "Monetary": "Chi tiêu (£)"}
      )
      fig_db.update_layout(plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19", font=dict(size=10), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
      st.plotly_chart(fig_db, use_container_width=True)

  with tab2:
    st.markdown(f"<h3 style='color: #00e5ff;'>BÁO CÁO ĐÓNG GÓP DOANH THU THEO PHÂN PHÚC ({selected_algorithm.upper()})</h3>", unsafe_allow_html=True)
    
    # Tổng hợp theo phân khúc kinh doanh
    revenue_summary = (
        rfm.groupby("Segment_Name")
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
            "Segment_Name": "Phân khúc khách hàng",
            "Customer_Count": "Số lượng KH (Người)",
            "Total_Revenue": "Tổng doanh thu (£)",
            "Revenue_Share(%)": "Tỷ trọng đóng góp (%)",
            "Avg_Recency": "Số ngày mua gần nhất (TB)",
            "Median_Recency": "Số ngày mua gần nhất (Median)",
            "Avg_Frequency": "Tần suất mua (TB)",
            "Median_Frequency": "Tần suất mua (Median)",
            "Avg_Monetary": "Chi tiêu KH (TB £)",
            "Median_Monetary": "Chi tiêu KH (Median £)",
        }
    )

    # Hiển thị bảng không bị cắt cột nhờ container_width
    st.dataframe(
        revenue_summary.style.format({
            "Tổng doanh thu (£)": "{:,.2f}",
            "Tỷ trọng đóng góp (%)": "{:.2f}%",
            "Số ngày mua gần nhất (TB)": "{:.1f}",
            "Số ngày mua gần nhất (Median)": "{:.1f}",
            "Tần suất mua (TB)": "{:.1f}",
            "Tần suất mua (Median)": "{:.1f}",
            "Chi tiêu KH (TB £)": "{:,.2f}",
            "Chi tiêu KH (Median £)": "{:,.2f}",
            "Số lượng KH (Người)": "{:,}",
        }),
        use_container_width=True,
    )

    # Biểu đồ phân bổ doanh thu có nhãn nhóm rõ ràng và chú thích cụ thể
    fig_rev = px.bar(
        revenue_summary,
        x="Phân khúc khách hàng",
        y="Tổng doanh thu (£)",
        text="Tỷ trọng đóng góp (%)",
        title=f"Biểu đồ phân bổ doanh thu theo phân khúc ({selected_algorithm})",
        template="plotly_dark",
        labels={"Tổng doanh thu (£)": "Tổng doanh thu (£)", "Phân khúc khách hàng": "Phân khúc"}
    )
    fig_rev.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_rev.update_layout(
        plot_bgcolor="#0b0f19",
        paper_bgcolor="#0b0f19",
        title_font=dict(color="#ffffff", size=16),
        xaxis=dict(title_font=dict(color="#ffffff")),
        yaxis=dict(title_font=dict(color="#ffffff"))
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Gợi ý hành động chiến lược cho từng nhóm (chưa kiểm chứng thực nghiệm)
    st.markdown("<h3 style='color: #00e5ff;'>GỢI Ý HÀNH ĐỘNG CHIẾN LƯỢC (CHƯA KIỂM CHỨNG THỰC NGHIỆM)</h3>", unsafe_allow_html=True)
    st.info("Lưu ý: Các đề xuất dưới đây dựa trên phân tích đặc tính RFM thực tế và cần được kiểm chứng thông qua A/B testing trước khi triển khai quy mô lớn.")
    
    for idx, row in revenue_summary.iterrows():
      seg_name = row["Phân khúc khách hàng"]
      share = row["Tỷ trọng đóng góp (%)"]
      count = row["Số lượng KH (Người)"]
      
      if "VIP" in seg_name:
        action = "Dự án chăm sóc đặc quyền, tặng voucher sinh nhật, dịch vụ hỗ trợ ưu tiên (VIP Support) để duy trì lòng trung thành."
      elif "trung thành" in seg_name:
        action = "Chương trình tích điểm thân thiết, giới thiệu sản phẩm chéo (Cross-selling) và upsell dòng sản phẩm cao cấp."
      elif "ngủ đông" in seg_name:
        action = "Triển khai chiến dịch Win-back (gửi email nhắc nhở, mã giảm giá kích hoạt lại lần mua tiếp theo sau thời gian vắng bóng)."
      else:
        action = "Theo dõi hành vi phát sinh và tối ưu hóa chi phí tiếp cận qua các kênh quảng cáo đại trà."
        
      st.markdown(f"* **{seg_name}** ({count:,} khách hàng, chiếm {share:.2f}% doanh thu): {action}")

  with tab3:
    st.markdown("<h3 style='color: #00e5ff;'>CHI TIẾT KHÁCH HÀNG & BỘ LỌC PHÂN KHÚC</h3>", unsafe_allow_html=True)
    
    # Bộ lọc theo phân khúc và tìm kiếm mã khách hàng
    col_f1, col_f2 = st.columns(2)
    with col_f1:
      search_query = st.text_input("🔍 Tìm kiếm theo Mã Khách Hàng (CustomerID):", "")
    with col_f2:
      unique_segments = ["Tất cả"] + list(rfm["Segment_Name"].unique())
      selected_seg_filter = st.selectbox("📂 Lọc theo phân khúc kinh doanh:", unique_segments)

    filtered_rfm = rfm.copy()
    if search_query:
      filtered_rfm = filtered_rfm[filtered_rfm["CustomerID"].astype(str).str.contains(search_query, na=False)]
    if selected_seg_filter != "Tất cả":
      filtered_rfm = filtered_rfm[filtered_rfm["Segment_Name"] == selected_seg_filter]

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