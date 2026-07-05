import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

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
<p style='text-align:center;'>PCA + KMeans + SVM | Dibuja un dígito (0-9)</p>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# INPUT
# ============================================================
st.subheader("✏️ Ingresa un dígito")

modo = st.radio("Modo de entrada", ["🎲 Ejemplo automático", "✏️ Dibujar"], horizontal=True, key="modo")

sample = None
canvas_key = "canvas_dibujo"

if modo == "🎲 Ejemplo automático":
    # 🔄 SEMILLA ALEATORIA DIFERENTE CADA VEZ
    # Usar el tiempo actual + un contador de sesión para variar
    if 'random_counter' not in st.session_state:
        st.session_state.random_counter = 0
    st.session_state.random_counter += 1
    
    # Semilla basada en contador para que cambie en cada interacción
    np.random.seed(None)  # Resetear semilla
    sample = np.random.randint(0, 256, size=784).astype(float)
    
    # Mostrar miniatura del ejemplo
    img_example = sample.reshape(28, 28)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(img_example, caption="Ejemplo generado", width=150, clamp=True)
    
    st.info(f"🎲 Ejemplo aleatorio generado (min: {sample.min():.0f}, max: {sample.max():.0f})")

else:  # Dibujar
    if CANVAS_AVAILABLE:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Dibuja aquí (mantén presionado el mouse):**")
            
            # Canvas con configuración mejorada para dibujo continuo
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",      # Negro sólido para relleno
                stroke_width=15,                     # Grosor ajustado
                stroke_color="#FFFFFF",               # Blanco para el trazo
                background_color="#000000",           # Fondo negro
                background_image=None,
                update_streamlit=True,                # Actualizar en tiempo real
                height=280,
                width=280,
                drawing_mode="freedraw",              # Modo dibujo libre
                point_display_radius=0,
                key=canvas_key,                       # Key fija para persistencia
            )
            
            # Botón para limpiar con key única
            if st.button("🗑️ Limpiar canvas", key="btn_limpiar"):
                # Forzar recarga con nueva key para el canvas
                st.session_state.canvas_key = st.session_state.get('canvas_key', 0) + 1
                st.rerun()
        
        with col2:
            st.markdown("**Procesado:**")
            
            if canvas_result.image_data is not None:
                # La imagen del canvas es RGBA, convertir a escala de grises
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img_gray = img.convert('L')
                
                # Redimensionar a 28x28
                img_28 = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img_28)
                
                # Invertir: canvas tiene fondo negro(0) y trazo blanco(255)
                # MNIST tiene fondo negro(0) y dígito blanco(255-0)
                # El canvas ya tiene esta configuración, solo normalizamos
                sample = img_array.flatten().astype(float)
                
                # Mostrar imagen procesada
                st.image(img_28, caption="28×28 procesado", width=150, clamp=True)
                
                # Estadísticas
                pixeles_activos = (sample > 30).sum()
                st.caption(f"Píxeles activos: {pixeles_activos}/784 | "
                          f"Intensidad promedio: {sample.mean():.1f}")
            else:
                st.warning("Dibuja algo en el canvas 👈")
                sample = np.zeros(784)
    else:
        st.error("⚠️ `streamlit-drawable-canvas` no está instalado.")
        st.code("pip install streamlit-drawable-canvas", language="bash")
        sample = np.zeros(784)

# ============================================================
# PREDICCIÓN
# ============================================================
if sample is not None:
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        predecir = st.button("🔍 Predecir dígito", type="primary", use_container_width=True)
    
    if predecir:
        # Normalizar
        sample_norm = sample / 255.0
        sample_reshaped = sample_norm.reshape(1, -1)
        
        # PCA
        sample_pca = pca.transform(sample_reshaped)
        
        # SVM
        model = svm_models["rbf"]
        pred = model.predict(sample_pca)[0]
        
        # KMeans
        cluster = kmeans.predict(sample_pca)[0]

        # ============================================================
        # RESULTADO
        # ============================================================
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                color: white;
                margin: 1rem 0;">
                
                <p style="font-size: 1.1rem; color: #e0e0e0; margin: 0;">🎯 Dígito Predicho</p>
                <p style="font-size: 6rem; font-weight: bold; margin: 0; color: #fff;">{pred}</p>
                <p style="font-size: 1rem; color: #e0e0e0; margin: 0.5rem 0;">SVM — Kernel RBF</p>
                <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 2rem;">
                <p style="font-size: 1.1rem; color: #e0e0e0; margin: 0;">
                    Cluster KMeans: <b style="color: #ffd700; font-size: 1.3rem;">{cluster}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Debug detallado
            with st.expander("🔍 Ver detalles técnicos"):
                st.write(f"**Input:** min={sample.min():.1f}, max={sample.max():.1f}, "
                        f"media={sample.mean():.1f}")
                st.write(f"**Normalizado:** min={sample_norm.min():.4f}, "
                        f"max={sample_norm.max():.4f}")
                st.write(f"**PCA:** shape={sample_pca.shape}, "
                        f"PC1={sample_pca[0,0]:.4f}, PC2={sample_pca[0,1]:.4f}")
                st.write(f"**Predicción:** dígito={pred}, cluster={cluster}")

# Footer
st.divider()
st.caption("Desarrollado con ❤️ | PCA + KMeans + SVM | MNIST Dataset")
