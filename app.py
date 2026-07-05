
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
    sample = np.random.rand(784)
else:
    sample = []
    for i in range(784):
        sample.append(st.number_input(f"Pixel {i}", 0.0, 255.0, 0.0))
    sample = np.array(sample)

# ============================================================
# BOTÓN PREDICCIÓN
# ============================================================
if st.button("🔍 Predecir dígito"):

    # Normalizar
    sample = sample / 255.0

    # PCA
    sample_pca = pca.transform([sample])

    # SVM (mejor kernel)
    model = svm_models["rbf"]
    pred = model.predict(sample_pca)[0]

    # Cluster KMeans
    cluster = kmeans.predict(sample_pca)[0]

    # ============================================================
    # RESULTADO
    # ============================================================
    st.markdown(f"""
    <div style="
        background:white;
        padding:2rem;
        border-radius:20px;
        text-align:center;
        box-shadow:0 10px 30px rgba(0,0,0,0.2);
        margin-top:20px;">
        
        <h2 style="color:#0d47a1;">Resultado</h2>
        <h1 style="font-size:4rem;">{pred}</h1>
        <p>Cluster KMeans: <b>{cluster}</b></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingresa los datos y presiona predecir")
