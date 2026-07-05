import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="MNIST Classifier IA",
    page_icon="🔢",
    layout="centered"
)

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")

PCA_PATH = MODEL_DIR / "pca_mnist.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_mnist.pkl"
SVM_PATH = MODEL_DIR / "svm_mnist.pkl"

@st.cache_resource
def cargar_modelos():
    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    svm_models = joblib.load(SVM_PATH)
    return pca, kmeans, svm_models

pca, kmeans, svm_models = cargar_modelos()

# DEBUG: Mostrar info de los modelos cargados
st.sidebar.header("🔧 Debug Info")
st.sidebar.write(f"PCA n_components: {pca.n_components_}")
st.sidebar.write(f"PCA n_features: {pca.n_features_in_}")
st.sidebar.write(f"KMeans n_features: {kmeans.n_features_in_}")
st.sidebar.write(f"KMeans clusters: {kmeans.n_clusters}")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<h1 style='text-align:center; color:#1976d2;'>🔢 Clasificador MNIST con IA</h1>
<p style='text-align:center;'>PCA + KMeans + SVM | Streamlit App</p>
""", unsafe_allow_html=True)

# ============================================================
# INPUT (28x28 PIXELS SIMULADO)
# ============================================================
st.subheader("✏️ Ingrese un vector de 784 píxeles")

modo = st.radio("Modo de entrada", ["Ejemplo automático", "Manual"])

if modo == "Ejemplo automático":
    # Valores entre 0-255 como en el dataset original MNIST
    sample = np.random.randint(0, 256, size=784).astype(float)
else:
    sample = []
    for i in range(784):
        sample.append(st.number_input(f"Pixel {i}", 0.0, 255.0, 0.0))
    sample = np.array(sample)

st.write(f"Sample shape: {sample.shape}")
st.write(f"Sample min: {sample.min():.1f}, max: {sample.max():.1f}")

# ============================================================
# BOTÓN PREDICCIÓN
# ============================================================
if st.button("🔍 Predecir dígito"):

    # ✅ NORMALIZAR: El PCA fue entrenado con X_norm = X / 255.0
    sample_norm = sample / 255.0
    
    # Asegurar shape correcto: (1, 784)
    sample_reshaped = sample_norm.reshape(1, -1)
    st.write(f"Shape para PCA: {sample_reshaped.shape}")
    
    # PCA: 784 features → 2 componentes
    try:
        sample_pca = pca.transform(sample_reshaped)
        st.write(f"Shape después de PCA: {sample_pca.shape}")
    except ValueError as e:
        st.error(f"Error en PCA: {e}")
        st.error(f"PCA espera {pca.n_features_in_} features, recibió {sample_reshaped.shape[1]}")
        st.stop()

    # SVM (mejor kernel: rbf)
    try:
        model = svm_models["rbf"]
        pred = model.predict(sample_pca)[0]
    except ValueError as e:
        st.error(f"Error en SVM: {e}")
        st.stop()

    # Cluster KMeans
    try:
        cluster = kmeans.predict(sample_pca)[0]
    except ValueError as e:
        st.error(f"Error en KMeans: {e}")
        st.error(f"KMeans espera {kmeans.n_features_in_} features, recibió {sample_pca.shape[1]}")
        st.stop()

    # ============================================================
    # RESULTADO
    # ============================================================
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-top: 20px;
        color: white;">
        
        <h2 style="color: #fff; margin-bottom: 0.5rem;">🎯 Resultado</h2>
        <h1 style="font-size: 6rem; margin: 0; color: #fff;">{pred}</h1>
        <p style="font-size: 1.2rem; color: #e0e0e0;">
            Dígito predicho por SVM (kernel RBF)
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 0;">
        <p style="font-size: 1rem; color: #e0e0e0;">
            Cluster KMeans: <b style="color: #ffd700;">{cluster}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingresa los datos y presiona predecir")
