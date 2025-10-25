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

# --------------------------
# CSS elegante para fondo y fuente
# --------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    /* Fuente global y fondo degradado suave */
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #FFA07A, #FF7F50, #6A5ACD, #483D8B);
        color: white;
    }

    /* Contenedores principales con transparencia y bordes redondeados */
    .css-1d391kg, .css-1v0mbdj { 
        background-color: rgba(0,0,0,0.25);
        border-radius: 12px;
        padding: 15px;
    }

    /* Títulos y subtítulos */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }

    /* Barra lateral con transparencia */
    .css-1d391kg .css-1v0mbdj {
        background-color: rgba(0,0,0,0.3);
        border-radius: 12px;
        padding: 15px;
    }

    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-track {
        background-color: rgba(0,0,0,0.1);
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar versión compatible de PyTorch y YOLOv5:
           
pip install torch==1.12.0 torchvision==0.13.0
pip install yolov5==7.0.9

        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

# Título y descripción
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# Cargar modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Sidebar
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
        picture = st.camera_input("Capturar imagen", key="camera")
        
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
                    st.subheader("Imagen con detecciones")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                
                with col2:
                    st.subheader("Objetos detectados")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Pie de página
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")
