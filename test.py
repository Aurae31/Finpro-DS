import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Paris Housing Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# CUSTOM UI THEME — PRODUCT DASHBOARD STYLE
# ==========================================
st.markdown("""
<style>

/* ===============================
   GLOBAL
=============================== */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Smooth transition for all components */
* {
    transition: all 0.25s ease-in-out;
}

/* ===============================
   SIDEBAR
=============================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* Sidebar radio */
div[role="radiogroup"] > label {
    background-color: transparent;
    border-radius: 12px;
    padding: 0.4rem 0.6rem;
}

div[role="radiogroup"] > label:hover {
    background-color: #1e293b;
}

/* ===============================
   METRIC CARDS
=============================== */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 1.2rem 1rem;
    border-radius: 20px;
    box-shadow: 0 12px 25px rgba(37, 99, 235, 0.25);
    animation: fadeInUp 0.6s ease forwards;
}

div[data-testid="metric-container"] label {
    color: #dbeafe;
}

div[data-testid="metric-container"] div {
    color: white;
}

/* ===============================
   ALERTS / STORYTELLING BOX
=============================== */
.stAlert {
    border-radius: 16px;
    border-left: 6px solid #2563eb;
    background-color: #f8fafc;
    animation: fadeIn 0.6s ease;
}

/* ===============================
   DATAFRAME
=============================== */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}

/* ===============================
   BUTTONS
=============================== */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    border-radius: 14px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}

button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 20px rgba(37, 99, 235, 0.35);
}

/* ===============================
   PLOTLY CHART CONTAINER
=============================== */
div[data-testid="stPlotlyChart"] {
    background-color: white;
    border-radius: 18px;
    padding: 0.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    animation: fadeInUp 0.6s ease;
}

/* ===============================
   ANIMATIONS
=============================== */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
            
/* Main app background responsive */
.stApp {
    background: linear-gradient(
        180deg,
        #f8fafc 0%,
        #f1f5f9 100%
    );
}

/* Mobile optimization */
@media (max-width: 768px) {
    .stApp {
        background: #f8fafc;
    }

    div[data-testid="metric-container"] {
        border-radius: 16px;
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Space bawah agar konten tidak ketutup footer */
.main > div {
    padding-bottom: 90px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("4. Paris Housing.csv")

df_raw = load_data()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🏠 Paris Housing ML")
st.sidebar.markdown("Prediksi Harga Properti Berbasis Data")

tab_selection = st.sidebar.radio(
    "Menu Utama",
    ["About", "Dashboard", "Modeling", "Machine Learning", "Prediction App", "Kontak"]
)

# ==========================================
# PAGE TRANSITION HANDLER
# ==========================================
if "last_tab" not in st.session_state:
    st.session_state.last_tab = tab_selection

transition_class = "fade-in"

if st.session_state.last_tab != tab_selection:
    transition_class = "fade-slide"
    st.session_state.last_tab = tab_selection

st.markdown(f"""
<style>
.fade-in {{
    animation: fadeIn 0.4s ease-in;
}}

.fade-slide {{
    animation: fadeSlide 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}}

@keyframes fadeSlide {{
    from {{
        opacity: 0;
        transform: translateY(16px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}
</style>

<div class="{transition_class}">
""", unsafe_allow_html=True)

# ==========================================
# TAB ABOUT
# ==========================================
if tab_selection == "About":
    st.title("🏠 Paris Housing Price Prediction")
    st.image(
        "https://i.pinimg.com/originals/75/15/9c/75159cce34357a305ae8db7cba1a5436.jpg",
        use_container_width=True
    )
    st.markdown("""
    Proyek ini bertujuan membangun **model prediksi harga rumah di Paris**
    menggunakan pendekatan **data-driven & interpretable machine learning**.
    """)

    st.subheader("📘 Tentang Dataset")

    st.markdown("""
    Dataset **Paris Housing** berisi informasi properti residensial di Paris
    yang dirancang untuk analisis harga dan pemodelan regresi.

    ### 🔹 Karakteristik Dataset
    - **Jumlah data:** Ribuan properti
    - **Target variabel:** `price`
    - **Tipe data:** Numerik & kategorikal
    - **Kondisi data:** Bersih, minim missing value

    ### 🔹 Contoh Fitur Penting
    - `squareMeters` → luas bangunan (driver utama harga)
    - `numberOfRooms` → kompleksitas properti
    - `floors` → struktur bangunan
    - `hasPool`, `hasYard` → fitur premium
    - `category` → kelas properti (Basic / Luxury)

    ### 🔹 Kegunaan Dataset
    Dataset ini ideal untuk:
    - Exploratory Data Analysis (EDA)
    - Regresi linear & regularized regression
    - Studi interpretabilitas model
    - Simulasi data dunia nyata
    """)
    st.dataframe(df_raw.head(), use_container_width=True)

# ==========================================
# TAB DASHBOARD
# ==========================================
elif tab_selection == "Dashboard":
    st.title("📊 Dashboard & Analytical Storytelling")

    st.markdown("""
<div style="
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 2.8rem;
    border-radius: 28px;
    color: white;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 40px rgba(37,99,235,0.35);
">
    <h1 style="font-size:2.6rem; margin-bottom:0.6rem;">
        🏠 Paris Housing Market Intelligence
    </h1>
    <p style="font-size:1.15rem; max-width:800px; line-height:1.6;">
        Interactive analytical dashboard untuk mengeksplorasi faktor utama harga properti
        di Paris menggunakan pendekatan <b>Exploratory Data Analysis</b> dan
        <b>Interpretable Machine Learning</b>.
    </p>
</div>
""", unsafe_allow_html=True)

    st.header("1️⃣ Data Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Properti", f"{len(df_raw):,}")
    c2.metric("Rata-rata Harga", f"€{df_raw['price'].mean():,.0f}")
    c3.metric("Median Harga", f"€{df_raw['price'].median():,.0f}")

    st.info("Dataset besar dan stabil → cocok untuk pendekatan regresi.")

    st.header("2️⃣ Automated EDA Summary")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns

    eda_summary = pd.DataFrame({
        "Missing (%)": df_raw[num_cols].isna().mean() * 100,
        "Mean": df_raw[num_cols].mean(),
        "Std": df_raw[num_cols].std(),
        "Skewness": df_raw[num_cols].skew()
    }).round(2)

    st.dataframe(eda_summary, use_container_width=True)

    st.header("3️⃣ Distribusi & Hubungan Variabel")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df_raw, x="price", nbins=50, marginal="box"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(df_raw.sample(1000, random_state=42), x="squareMeters", y="price", opacity=0.6), use_container_width=True)

    st.success("Hubungan linear kuat → Linear / Ridge Regression relevan.")

    st.header("4️⃣ Feature Importance (Correlation)")
    corr = df_raw[num_cols].corr()["price"].abs().sort_values(ascending=False).reset_index()
    corr.columns = ["Feature", "Abs Correlation"]
    st.plotly_chart(px.bar(corr.head(10), x="Abs Correlation", y="Feature", orientation="h"), use_container_width=True)

    st.header("5️⃣ Insight → Keputusan Modeling")
    st.success("""
    **Keputusan Modeling:**
    - StandardScaler
    - Linear vs Ridge vs Lasso
    - Ridge diprioritaskan untuk stabilitas dunia nyata
    """)

# ==========================================
# TAB MODELING
# ==========================================
elif tab_selection == "Modeling":
    st.title("🏫 Dokumentasi Teknis Model")
    st.markdown("Berikut adalah langkah-langkah detail (Step-by-Step) pengerjaan model machine learning beserta kode implementasinya.")

    # =====================================
    # STEP 1: IMPORT & EKSPLORASI DATA
    # =====================================
    st.header("Step 1: Import & Eksplorasi Data")

    df = pd.read_csv("4. Paris Housing.csv")

    numbers = df.select_dtypes(include=['number']).columns
    categories = df.select_dtypes(exclude=['number']).columns

    st.code("""
df = pd.read_csv('4. Paris Housing.csv')
numbers = df.select_dtypes(include=['number']).columns
categories = df.select_dtypes(exclude=['number']).columns
""", language="python")

    st.subheader("📌 Preview Dataset")
    st.dataframe(df.head())

    st.write("Shape Dataset:", df.shape)
    st.write("Kolom Numerik:", list(numbers))
    st.write("Kolom Kategorikal:", list(categories))

    # =====================================
    # STEP 2: DATA CLEANING
    # =====================================
    st.header("Step 2: Data Cleaning (Outlier & Encoding)")

    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]

    le = LabelEncoder()
    df_clean["category"] = le.fit_transform(df_clean["category"])

    st.code("""
# IQR Outlier Removal
df_clean = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]

# Label Encoding
df_clean['category'] = LabelEncoder().fit_transform(df_clean['category'])
""", language="python")

    st.write("Jumlah data sebelum cleaning:", df.shape[0])
    st.write("Jumlah data setelah cleaning:", df_clean.shape[0])

    # =====================================
    # STEP 3: FEATURE SELECTION & VIF
    # =====================================
    st.header("Step 3: Feature Selection & Multikolinearitas (VIF)")

    X = df_clean.drop("price", axis=1)
    y = df_clean["price"]

    scaler_vif = StandardScaler()
    X_scaled_vif = scaler_vif.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_vif, columns=X.columns)

    vif_df = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [
            variance_inflation_factor(X_scaled_df.values, i)
            for i in range(X_scaled_df.shape[1])
        ]
    })

    st.code("""
vif_df = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [
        variance_inflation_factor(X_scaled_df.values, i)
        for i in range(X_scaled_df.shape[1])
    ]
})
""", language="python")

    st.subheader("📌 VIF Result")
    st.dataframe(vif_df)

    # =====================================
    # STEP 4: TRAIN TEST SPLIT & SCALING
    # =====================================
    st.header("Step 4: Splitting & Scaling")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.code("""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
""", language="python")

    st.write("Shape X_train:", X_train.shape)
    st.write("Shape X_test:", X_test.shape)

    # =====================================
    # STEP 5: MODELING & HYPERPARAMETER
    # =====================================
    st.header("Step 5: Modeling & Hyperparameter Tuning")

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    ridge = GridSearchCV(
        Ridge(),
        {"alpha": np.logspace(-3, 3, 20)},
        cv=5
    )
    ridge.fit(X_train_scaled, y_train)

    lasso = GridSearchCV(
        Lasso(max_iter=10000),
        {"alpha": np.logspace(-3, 3, 20)},
        cv=5
    )
    lasso.fit(X_train_scaled, y_train)

    st.subheader("📌 Koefisien Linear Regression")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lr.coef_
    }).sort_values(by="Coefficient", ascending=False)

    st.dataframe(coef_df)

    st.write("Best Alpha Ridge:", ridge.best_params_)
    st.write("Best Alpha Lasso:", lasso.best_params_)

    # =====================================
    # STEP 6: EVALUASI MODEL
    # =====================================
    st.header("Step 6: Evaluasi & Kesimpulan Model")

    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }

    eval_df = pd.DataFrame([
        evaluate(y_test, y_pred_lr),
        evaluate(y_test, ridge.predict(X_test_scaled)),
        evaluate(y_test, lasso.predict(X_test_scaled))
    ], index=["Linear Regression", "Ridge Regression", "Lasso Regression"])

    st.subheader("📊 Hasil Evaluasi")
    st.dataframe(eval_df)

    st.success("✅ Seluruh pipeline berhasil dijalankan dan ditampilkan dengan output nyata.")

# ==========================================
# TAB MACHINE LEARNING
# ==========================================
elif tab_selection == "Machine Learning":
    st.title("🤖 Machine Learning Workbench")
    np.random.seed(42)

    noise = st.slider("Noise Level", 0.0, 0.5, 0.15, 0.01)
    df_ml = df_raw.copy()

    if "category" in df_ml.columns:
        le = LabelEncoder()
        df_ml["category"] = le.fit_transform(df_ml["category"])

    df_ml["price"] += np.random.normal(0, df_ml["price"].std(), len(df_ml)) * noise

    X = df_ml.drop("price", axis=1)
    y = df_ml["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_choice = st.selectbox(
        "Pilih Model",
        ["Linear Regression", "Ridge Regression", "Lasso Regression"]
    )

    alphas = np.logspace(-3, 3, 20)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Ridge Regression":
        model = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5)
    else:
        model = GridSearchCV(Lasso(), {"alpha": alphas}, cv=5)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(y_test, preds):,.0f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):,.0f}")
    c3.metric("R²", f"{r2_score(y_test, preds):.4f}")

    # Visualisasi
    fig_eval = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Prediksi vs Aktual", opacity=0.5)
    fig_eval.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_eval, use_container_width=True)

# ==========================================
# TAB PREDICTION APP 
# ==========================================
elif tab_selection == "Prediction App":
    st.title("🔮 Demo Price Prediction")
    st.caption("⚠️ Model ini menggunakan preprocessing yang sama dengan training (scaling konsisten).")

    # ===============================
    # PREPARE DATA
    # ===============================
    X_demo = df_raw.drop("price", axis=1)
    y_demo = df_raw["price"]

    # Encode categorical
    if "category" in X_demo.columns:
        le = LabelEncoder()
        X_demo["category"] = le.fit_transform(X_demo["category"])

    # ===============================
    # SCALING + TRAIN MODEL
    # ===============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)

    model = LinearRegression()
    model.fit(X_scaled, y_demo)

    # ===============================
    # USER INPUT
    # ===============================
    sqm = st.number_input("Luas Bangunan (m²)", 50, 100000, 500)
    rooms = st.number_input("Jumlah Kamar", 1, 50, 5)
    floors = st.number_input("Jumlah Lantai", 1, 20, 2)

    # ===============================
    # PREPARE SAMPLE
    # ===============================
    sample = X_demo.mean().to_frame().T
    sample["squareMeters"] = sqm
    sample["numberOfRooms"] = rooms
    sample["floors"] = floors

    sample_scaled = scaler.transform(sample)

    if st.button("Prediksi Harga"):
        result = model.predict(sample_scaled)
        st.success(f"💰 Estimasi Harga: €{result[0]:,.2f}")

# ==========================================
# TAB Kontak 
# ==========================================

elif tab_selection == "Kontak":
    st.title("✉️ Kontak")

    components.html("""
    <div style="
        background-color: white;
        padding: 32px;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        max-width: 600px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        font-family: Inter, system-ui, sans-serif;
    ">
        <h2 style="margin-bottom:8px; color:#1f2937;">
            👋 Rafly Azmi
        </h2>

        <p style="margin-bottom:24px; color:#4b5563; font-size:15px;">
            Machine Learning & Data Science Enthusiast<br>
            Paris Housing Price Prediction Project
        </p>

        <p style="font-size:16px; line-height:1.9;">
            👤 <b>Role</b><br>
            Machine Learning / Data Science Portfolio
        </p>

        <p style="font-size:16px; line-height:1.9;">
            🏠 <b>Project</b><br>
            Paris Housing Machine Learning
        </p>

        <p style="font-size:16px; line-height:1.9;">
            📧 <b>Email</b><br>
            <a href="mailto:31raflyazmi@gmail.com"
               style="color:#2563eb; text-decoration:none;">
               31raflyazmi@gmail.com
            </a>
        </p>

        <p style="font-size:16px; line-height:1.9;">
            💼 <b>LinkedIn</b><br>
            <a href="https://linkedin.com/in/raflyazmi" target="_blank"
               style="color:#2563eb; text-decoration:none; font-weight:600;">
               linkedin.com/in/raflyazmi
            </a>
        </p>
    </div>
    """, height=520)

# ==========================================
# STICKY GLOBAL FOOTER
# ==========================================
components.html("""
<style>
#sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: white;
    border-top: 1px solid #e5e7eb;
    z-index: 9999;
    box-shadow: 0 -8px 20px rgba(0,0,0,0.05);
}

#sticky-footer .container {
    max-width: 1200px;
    margin: auto;
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: Inter, system-ui, sans-serif;
    font-size: 14px;
    color: #6b7280;
}

#sticky-footer a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 500;
}

#sticky-footer a:hover {
    text-decoration: underline;
}
</style>

<div id="sticky-footer">
    <div class="container">
        <div>
            © 2026 <b>Rafly Azmi</b> · Paris Housing ML
        </div>

        <div style="display:flex; gap:18px;">
            <a href="mailto:31raflyazmi@gmail.com">📧 Email</a>
            <a href="https://linkedin.com/in/raflyazmi" target="_blank">💼 LinkedIn</a>
        </div>
    </div>
</div>
""", height=120)