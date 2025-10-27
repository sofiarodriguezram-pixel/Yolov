import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# --- ESTILOS PERSONALIZADOS ---
page_bg = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fefefe, #f3f8ff, #eaf3fa);
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background-color: #f9fbfd;
    }
    .image-frame {
        border: 3px solid #d6e0f5;
        border-radius: 15px;
        padding: 10px;
        background-color: white;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1b3b6f;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
        except Exception:
            st.warning("Intentando método alternativo de carga...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instala una versión compatible de PyTorch y YOLOv5:
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta.
        """)
        return None

# --- INTERFAZ PRINCIPAL ---
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("Parámetros")
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    main_container = st.container()
    with main_container:
        picture = st.camera_input("📸 Capturar imagen", key="camera")
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🖼️ Imagen con detecciones")
                    results.render()
                    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.subheader("📋 Objetos detectados")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[idx] = category_count.get(idx, 0) + 1

                    data = []
                    for cat, count in category_count.items():
                        label = label_names[cat]
                        confidence = scores[categories == cat].mean().item() if len(scores) > 0 else 0
                        data.append({"Categoría": label, "Cantidad": count, "Confianza promedio": f"{confidence:.2f}"})

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba reduciendo el umbral de confianza.")

            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")

st.markdown("---")
st.caption("**Acerca de la aplicación**: Desarrollada con Streamlit y PyTorch. Utiliza YOLOv5 para detección de objetos en tiempo real.")
