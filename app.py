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

st.divider()

# ============================================================
# INPUT (28x28 PIXELS SIMULADO)
# ============================================================
st.subheader("✏️ Ingrese un vector de 784 píxeles")

modo = st.radio("Modo de entrada", ["Ejemplo automático", "Manual"], horizontal=True)

if modo == "Ejemplo automático":
    # Valores entre 0-255 como en el dataset original MNIST
    sample = np.random.randint(0, 256, size=784).astype(float)
    st.info(f"🎲 Ejemplo generado automáticamente (min: {sample.min():.0f}, max: {sample.max():.0f})")
else:
    with st.expander("📝 Ingresar píxeles manualmente"):
        sample = []
        cols = st.columns(28)
        for i in range(784):
            col_idx = i % 28
            with cols[col_idx]:
                sample.append(st.number_input(f"P{i}", 0.0, 255.0, 0.0, key=f"pixel_{i}", label_visibility="collapsed"))
        sample = np.array(sample)

# ============================================================
# BOTÓN PREDICCIÓN
# ============================================================
if st.button("🔍 Predecir dígito", type="primary", use_container_width=True):

    # Normalizar: El PCA fue entrenado con X_norm = X / 255.0
    sample_norm = sample / 255.0
    sample_reshaped = sample_norm.reshape(1, -1)
    
    # PCA: 784 features → 2 componentes
    sample_pca = pca.transform(sample_reshaped)
    
    # SVM (mejor kernel: rbf)
    model = svm_models["rbf"]
    pred = model.predict(sample_pca)[0]
    
    # Cluster KMeans
    cluster = kmeans.predict(sample_pca)[0]

    # ============================================================
    # RESULTADO
    # ============================================================
    st.divider()
    
    # Tarjeta de resultado
    with st.container():
        st.markdown("""
        <style>
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            color: white;
            margin: 1rem 0;
        }
        .result-digit {
            font-size: 6rem;
            font-weight: bold;
            margin: 0;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .result-label {
            font-size: 1.1rem;
            color: #e0e0e0;
            margin: 0.5rem 0;
        }
        .result-cluster {
            font-size: 1.2rem;
            color: #ffd700;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">🎯 Dígito Predicho</p>
            <p class="result-digit">{pred}</p>
            <p class="result-label">Clasificador SVM — Kernel RBF</p>
            <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 2rem;">
            <p class="result-label">Cluster KMeans: <span class="result-cluster">{cluster}</span></p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👆 Selecciona un modo de entrada y presiona **Predecir dígito**")

# Footer
st.divider()
st.caption("Desarrollado con ❤️ | PCA + KMeans + SVM | MNIST Dataset")
