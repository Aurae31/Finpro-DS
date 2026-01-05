import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="Paris Housing Minpro", layout="wide")

# --- LOAD DATA (Berdasarkan Notebook) ---
@st.cache_data
def load_data():
    df = pd.read_csv('4. Paris Housing.csv')
    # Preprocessing: Encoding kategori sesuai notebook
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigasi Project")
tab_selection = st.sidebar.radio("Pilih Tab:", ["About", "Dashboard", "Machine Learning", "Prediction App", "Kontak"])

# ==========================================
# TAB 1: ABOUT
# ==========================================
if tab_selection == "About":
    st.title("🏠 About Dataset - Paris Housing")
    
    # Menambahkan gambar dari URL yang Anda berikan
    st.image("https://i.pinimg.com/originals/75/15/9c/75159cce34357a305ae8db7cba1a5436.jpg", 
             caption="Paris Housing Architecture", 
             use_container_width=True)

    st.markdown("""
    ### Deskripsi Proyek
    Berdasarkan notebook **Minpro Data Housing**, proyek ini bertujuan untuk memprediksi harga properti di Paris menggunakan berbagai model regresi. 
    Dataset ini mencakup berbagai spesifikasi teknis rumah seperti luas, jumlah lantai, hingga keberadaan fasilitas mewah.

    **Struktur Data:**
    - **Fitur Numerik:** Luas tanah, jumlah kamar, lantai, tahun pembuatan, dll.
    - **Fitur Kategorikal:** Kategori (Basic/Luxury) yang telah di-encode.
    - **Target:** `price` (Harga).
    """)
    st.write("#### Sampel Data Teratas:")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# TAB 2: DASHBOARD (Plotly Express)
# ==========================================
elif tab_selection == "Dashboard":
    st.title("📊 Data Dashboard (Interaktif)")
    
    # KPI Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel", f"{len(df):,}")
    col2.metric("Median Harga", f"€{df['price'].median():,.0f}")
    col3.metric("Rata-rata Luas", f"{df['squareMeters'].mean():,.1f} m²")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribusi Harga")
        fig_hist = px.histogram(df, x="price", nbins=40, marginal="violin", 
                                title="Distribusi Variabel Target (Price)",
                                color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("Hubungan Luas vs Harga")
        fig_scatter = px.scatter(df.sample(2500), x="squareMeters", y="price", 
                                 color="category", opacity=0.6,
                                 title="Korelasi Luas Tanah terhadap Harga",
                                 trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Korelasi Fitur (Heatmap)")
        # Hanya kolom numerik untuk korelasi
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                             title="Matriks Korelasi Numerik",
                             color_continuous_scale='Blues')
        st.plotly_chart(fig_corr, use_container_width=True)

    with c4:
        st.subheader("Harga Berdasarkan Kategori")
        fig_box = px.box(df, x="category", y="price", color="category",
                         title="Perbandingan Harga: Basic vs Luxury")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING (Logika Notebook)
# ==========================================
elif tab_selection == "Machine Learning":
    st.title("🤖 Pemodelan Machine Learning")
    st.write("Proses ini mengikuti langkah-langkah di notebook: Split, Scaling, dan Hyperparameter Tuning.")

    # Persiapan Data
    X = df.drop(['price', 'category'], axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Pilihan Model
    model_opt = st.selectbox("Pilih Model untuk Ditampilkan:", ["Linear Regression", "Ridge (Tuned)", "Lasso (Tuned)"])
    
    # Logika tuning alpha dari notebook (np.logspace)
    alphas = np.logspace(-3, 3, 20)

    if model_opt == "Linear Regression":
        model = LinearRegression().fit(X_train_s, y_train)
    elif model_opt == "Ridge (Tuned)":
        grid = GridSearchCV(Ridge(), {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train_s, y_train)
        model = grid.best_estimator_
        st.write(f"Best Alpha Ridge: `{grid.best_params_['alpha']:.4f}`")
    else:
        grid = GridSearchCV(Lasso(), {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train_s, y_train)
        model = grid.best_estimator_
        st.write(f"Best Alpha Lasso: `{grid.best_params_['alpha']:.4f}`")

    y_pred = model.predict(X_test_s)

    # Evaluasi
    eval1, eval2, eval3 = st.columns(3)
    eval1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
    eval2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    eval3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    # Plot Hasil Prediksi
    fig_res = px.scatter(x=y_test, y=y_pred, labels={'x': 'Nilai Asli', 'y': 'Prediksi'},
                         title="Visualisasi Akurasi Prediksi", opacity=0.4)
    fig_res.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_res, use_container_width=True)

# ==========================================
# TAB 4: PREDICTION APP
# ==========================================
elif tab_selection == "Prediction App":
    st.title("🔮 Kalkulator Harga Rumah")
    
    with st.form("form_pred"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            sqm = st.number_input("Luas (squareMeters)", 100, 100000, 50000)
            rooms = st.number_input("Jumlah Kamar", 1, 100, 5)
            floors = st.number_input("Lantai", 1, 100, 2)
            year = st.number_input("Tahun (made)", 1990, 2025, 2015)
        with col_in2:
            yard = st.selectbox("Halaman?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
            pool = st.selectbox("Kolam Renang?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
            luxury = st.selectbox("Kategori?", [0, 1], format_func=lambda x: "Luxury" if x==1 else "Basic")
            newbuilt = st.selectbox("Bangunan Baru?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")

        btn = st.form_submit_button("Prediksi Harga Sekarang")

    if btn:
        # Simple model untuk prediksi di app
        X_final = df.drop(['price', 'category'], axis=1)
        y_final = df['price']
        predictor = LinearRegression().fit(X_final, y_final)
        
        # Urutan kolom disesuaikan dengan dataset asli
        # (Beberapa fitur default menggunakan nilai rata-rata dataset)
        user_data = [[sqm, rooms, yard, pool, floors, 50000, 5, 1, year, newbuilt, 1, 500, 500, 1, 1, 1, luxury]]
        prediction = predictor.predict(user_data)
        
        st.success(f"### Estimasi Harga Properti: €{prediction[0]:,.2f}")

# ==========================================
# TAB 5: KONTAK
# ==========================================
elif tab_selection == "Kontak":
    st.title("✉️ Hubungi Saya")
    st.markdown("""
    **Project Developer:** [Nama Anda]  
    Project ini dikembangkan untuk memenuhi tugas **Mini Project Data Science**.

    - 📧 **Email:** developer@example.com
    - 💼 **LinkedIn:** [linkedin.com/in/username](https://linkedin.com)
    - 📁 **GitHub:** [github.com/username](https://github.com)
    """)
    st.balloons()