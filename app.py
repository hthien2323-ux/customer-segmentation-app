import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="E-commerce Customer Segmentation",
    page_icon="⚡",
    layout="wide"
)

# --- CSS TÙY CHỈNH: NỀN ĐEN, CHỮ CỰC TO, TÔNG MÀU LẠNH KHÔNG PASTEL, KHÔNG ICON ---
st.markdown("""
    <style>
        /* Nền toàn bộ ứng dụng màu đen tuyền */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        /* Tiêu đề chính siêu to, màu lạnh sắc nét, phát sáng */
        .main-header {
            font-size: 3.8rem;
            color: #00FFFF;
            font-weight: 900;
            text-align: center;
            letter-spacing: 2px;
            margin-top: -20px;
            margin-bottom: 5px;
            text-shadow: 0px 0px 30px rgba(0, 255, 255, 0.5);
        }
        .sub-header {
            text-align: center;
            color: #93C5FD;
            font-size: 1.4rem;
            margin-bottom: 35px;
            font-weight: 700;
        }
        
        /* Tiêu đề các mục: Cỡ chữ rất to, in đậm, tone lạnh rõ ràng, KHÔNG ICON */
        .sec-cyan {
            color: #00FFFF;
            font-weight: 900;
            font-size: 2.1rem;
            border-bottom: 3px solid #00FFFF;
            padding-bottom: 8px;
            margin-top: 25px;
            text-transform: uppercase;
        }
        .sec-blue {
            color: #3B82F6;
            font-weight: 900;
            font-size: 2.1rem;
            border-bottom: 3px solid #3B82F6;
            padding-bottom: 8px;
            margin-top: 25px;
            text-transform: uppercase;
        }
        .sec-purple {
            color: #A855F7;
            font-weight: 900;
            font-size: 2.1rem;
            border-bottom: 3px solid #A855F7;
            padding-bottom: 8px;
            margin-top: 25px;
            text-transform: uppercase;
        }
        
        /* Metric số liệu to và rõ */
        div[data-testid="stMetricValue"] {
            font-size: 2.6rem;
            font-weight: 900;
            color: #00FFFF;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1.2rem;
            font-weight: 700;
            color: #93C5FD;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề không icon
st.markdown('<p class="main-header">PHÂN KHÚC KHÁCH HÀNG TMĐT</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống phân tích hành vi khách hàng nâng cao kết hợp Học máy & Chỉ số RFM</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### **Cấu hình hệ thống**")
uploaded_file = st.sidebar.file_uploader("Tải lên file Online_Retail.csv", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file, encoding='latin1')
    else:
        try:
            return pd.read_csv("Online_Retail.csv", encoding='latin1')
        except:
            return None

df = load_data(uploaded_file)

if df is None:
    st.warning("Vui lòng đặt file `Online_Retail.csv` cùng thư mục với `app.py` hoặc tải lên ở thanh bên trái!")
    st.stop()

# --- XỬ LÝ DỮ LIỆU & RFM ---
with st.spinner("Đang tính toán chỉ số RFM và xử lý dữ liệu..."):
    try:
        df.columns = [str(col).strip().lower() for col in df.columns]

        id_col = [c for c in df.columns if 'customer' in c][0]
        date_col = [c for c in df.columns if 'date' in c][0]
        invoice_col = [c for c in df.columns if 'invoice' in c and ('no' in c or 'code' in c)][0]
        qty_col = [c for c in df.columns if 'quantity' in c][0]
        price_col = [c for c in df.columns if 'price' in c][0]

        df = df.dropna(subset=[id_col])
        df['totalprice'] = pd.to_numeric(df[qty_col], errors='coerce') * pd.to_numeric(df[price_col], errors='coerce')
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['totalprice', date_col])

        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby(id_col)
            .agg({
                date_col: lambda x: (snapshot_date - x.max()).days,
                invoice_col: "count",
                "totalprice": "sum"
            })
            .reset_index()
        )

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        st.stop()

# --- SIDEBAR: THUẬT TOÁN ---
st.sidebar.markdown("---")
st.sidebar.markdown("### **Thuật toán Phân cụm**")
model_choice = st.sidebar.selectbox("Chọn thuật toán", ["K-Means", "Hierarchical Clustering", "DBSCAN"])

scaler = StandardScaler()
X = rfm[["Recency", "Frequency", "Monetary"]]
X_scaled = scaler.fit_transform(X)

if model_choice == "K-Means":
    n_clusters = st.sidebar.slider("Số lượng cụm (K)", 2, 6, 4)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = model.fit_predict(X_scaled)

elif model_choice == "Hierarchical Clustering":
    n_clusters = st.sidebar.slider("Số lượng cụm (K)", 2, 6, 4)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    rfm['Cluster'] = model.fit_predict(X_scaled)

else:
    eps = st.sidebar.slider("Eps", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 3, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    rfm['Cluster'] = model.fit_predict(X_scaled)

rfm['Cluster'] = rfm['Cluster'].astype(str)

# --- HIỂN THỊ METRIC TỔNG QUAN ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tổng khách hàng", f"{rfm.shape[0]:,}")
m2.metric("Tổng doanh thu", f"${rfm['Monetary'].sum():,.0f}")
m3.metric("Tần suất mua TB", f"{rfm['Frequency'].mean():.1f} lần")
m4.metric("Số cụm phân tách", f"{rfm['Cluster'].nunique()}")

st.markdown("<br>", unsafe_allow_html=True)

# --- BỐ CỤC TABS (Không icon, màu sắc lạnh rõ ràng) ---
tab1, tab2, tab3 = st.tabs(["Trực quan hóa và Phân tích", "Thống kê chi tiết cụm", "Dữ liệu khách hàng"])

with tab1:
    st.markdown('<p class="sec-cyan">Biểu đồ Tương tác & Tỷ lệ Phân khúc</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.markdown("**Biểu đồ phân tán (Recency vs Monetary)**")
        fig = px.scatter(
            rfm, 
            x="Recency", 
            y="Monetary", 
            color="Cluster",
            size="Frequency",
            hover_data=["CustomerID", "Frequency"],
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Dark24,
            height=480
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), 
            paper_bgcolor='#000000', 
            plot_bgcolor='#000000',
            font=dict(color="#FFFFFF", size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_r:
        st.markdown("**Tỷ lệ phân bổ các cụm**")
        cluster_counts = rfm['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(
            cluster_counts, 
            values='Count', 
            names='Cluster', 
            hole=0.4, 
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), 
            height=450, 
            paper_bgcolor='#000000', 
            plot_bgcolor='#000000',
            font=dict(color="#FFFFFF", size=14)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.markdown('<p class="sec-blue">Bảng Tổng hợp Chỉ số Trung bình theo Phân khúc</p>', unsafe_allow_html=True)
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    cluster_summary['Customer_Count'] = rfm.groupby('Cluster')['CustomerID'].count().values
    
    cluster_summary = cluster_summary.rename(columns={
        'Recency': 'Recency TB (Ngày)',
        'Frequency': 'Frequency TB (Lần)',
        'Monetary': 'Monetary TB ($)',
        'Customer_Count': 'Số lượng KH'
    })
    st.dataframe(cluster_summary.style.format({
        'Recency TB (Ngày)': '{:.1f}',
        'Frequency TB (Lần)': '{:.1f}',
        'Monetary TB ($)': '{:,.2f}',
        'Số lượng KH': '{:,}'
    }), use_container_width=True)

with tab3:
    st.markdown('<p class="sec-purple">Danh sách Chi tiết RFM của Khách hàng</p>', unsafe_allow_html=True)
    st.dataframe(rfm, use_container_width=True)