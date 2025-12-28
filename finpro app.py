import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="Paris Housing Interactive Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('4. Paris Housing.csv')
    le = LabelEncoder()
    # Simpan kategori asli untuk dashboard, dan kategori encoded untuk machine learning
    df['category_encoded'] = le.fit_transform(df['category'])
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigasi")
tab_selection = st.sidebar.radio("Pilih Tab:", ["About", "Dashboard", "Machine Learning", "Prediction App", "Kontak"])

# ==========================================
# TAB 1: ABOUT
# ==========================================
if tab_selection == "About":
    st.title("🏠 About Dataset")
    st.markdown("""
    ### Paris Housing Price Dataset
    Dataset ini memberikan gambaran komprehensif mengenai pasar properti di Paris. 
    Melalui aplikasi ini, kita akan mengeksplorasi bagaimana fitur fisik bangunan menentukan nilai pasar.
    
    **Variabel dalam Dataset:**
    - **Dimensi:** `squareMeters`, `floors`, `basement`, `attic`, `garage`.
    - **Fasilitas:** `hasPool`, `hasYard`, `hasStormProtector`, `hasStorageRoom`.
    - **Kualitas:** `isNewBuilt`, `category` (Basic/Luxury).
    - **Target:** `price` (dalam Euro).
    """)
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# TAB 2: DASHBOARD (Menggunakan Plotly Express)
# ==========================================
elif tab_selection == "Dashboard":
    st.title("📊 Interactive Dashboard")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Rata-rata Harga", f"€{df['price'].mean():,.0f}")
    col3.metric("Rata-rata Luas", f"{df['squareMeters'].mean():,.0f} m²")
    col4.metric("% Luxury", f"{(df['category'] == 'Luxury').mean()*100:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribusi Harga Rumah")
        fig_hist = px.histogram(df, x="price", nbins=50, 
                                marginal="box", # Menambahkan boxplot di atas histogram
                                title="Sebaran Harga (Euro)",
                                color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Hubungan Luas vs Harga")
        fig_scatter = px.scatter(df.sample(2000), x="squareMeters", y="price", 
                                 color="category", 
                                 trendline="ols", # Menambahkan garis regresi
                                 title="Luas (m²) vs Harga",
                                 hover_data=['numberOfRooms'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Matriks Korelasi (Heatmap)")
        corr = df.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", 
                             color_continuous_scale='RdBu_r',
                             title="Korelasi Antar Fitur Numerik")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_right2:
        st.subheader("Harga per Kategori & Kolam Renang")
        fig_box = px.box(df, x="category", y="price", 
                         color="hasPool",
                         notched=True,
                         title="Perbandingan Harga Berdasarkan Fasilitas")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING
# ==========================================
elif tab_selection == "Machine Learning":
    st.title("🤖 Machine Learning Workbench")
    
    # Persiapan Data
    X = df.drop(['price', 'category'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_type = st.selectbox("Pilih Algoritma Regresi:", ["Linear Regression", "Ridge", "Lasso"])
    
    if model_type == "Linear Regression":
        model = LinearRegression().fit(X_train_scaled, y_train)
    elif model_type == "Ridge":
        alpha = st.slider("Alpha (Regularization strength)", 0.1, 100.0, 1.0)
        model = Ridge(alpha=alpha).fit(X_train_scaled, y_train)
    else:
        alpha = st.slider("Alpha (Regularization strength)", 0.1, 100.0, 1.0)
        model = Lasso(alpha=alpha).fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    
    # Evaluasi Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    c3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

    # Plot Hasil Prediksi vs Aktual
    fig_res = px.scatter(x=y_test, y=y_pred, 
                         labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                         title=f"Actual vs Predicted ({model_type})",
                         opacity=0.5)
    fig_res.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_res, use_container_width=True)

# ==========================================
# TAB 4: PREDICTION APP
# ==========================================
elif tab_selection == "Prediction App":
    st.title("🔮 Estimator Harga Rumah")
    
    with st.expander("Klik untuk Input Spesifikasi Rumah", expanded=True):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            sqm = st.number_input("Luas Tanah (m²)", 10, 100000, 5000)
            rooms = st.number_input("Jumlah Kamar", 1, 100, 3)
            floors = st.number_input("Jumlah Lantai", 1, 100, 1)
            year = st.number_input("Tahun Dibuat", 1900, 2025, 2020)
        with col_in2:
            yard = st.checkbox("Memiliki Halaman")
            pool = st.checkbox("Memiliki Kolam Renang")
            newbuilt = st.checkbox("Bangunan Baru")
            luxury = st.checkbox("Kategori Mewah (Luxury)")

    if st.button("Hitung Estimasi Harga"):
        # Training model sederhana untuk kebutuhan prediksi cepat di tab ini
        X_p = df.drop(['price', 'category'], axis=1)
        y_p = df['price']
        predictor = LinearRegression().fit(X_p, y_p)
        
        # Mapping input (mengikuti urutan kolom X_p)
        # Urutan: squareMeters, numberOfRooms, hasYard, hasPool, floors, cityCode, cityPartRange, numPrevOwners, made, isNewBuilt, hasStormProtector, basement, attic, garage, hasStorageRoom, hasGuestRoom, category_encoded
        # (Beberapa nilai default diatur ke median agar tidak mempengaruhi hasil drastis)
        sample_input = [[sqm, rooms, int(yard), int(pool), floors, 50000, 5, 1, year, int(newbuilt), 0, 1000, 1000, 1, 1, 1, int(luxury)]]
        hasil = predictor.predict(sample_input)
        
        st.success(f"### Estimasi Harga Properti: €{hasil[0]:,.2f}")
        st.info("Catatan: Prediksi ini didasarkan pada model Linear Regression yang dilatih pada dataset Paris Housing.")

# ==========================================
# TAB 5: KONTAK
# ==========================================
elif tab_selection == "Kontak":
    st.title("✉️ Informasi Kontak")
    st.markdown("""
    Aplikasi ini dikembangkan sebagai bagian dari **Final Project Data Science**. 
    Silakan hubungi saya untuk diskusi lebih lanjut:
    
    - 📧 **Email:** developer@example.com
    - 🔗 **LinkedIn:** [linkedin.com/in/namaanda](#)
    - 💻 **GitHub:** [github.com/namaanda](#)
    """)
    st.balloons()