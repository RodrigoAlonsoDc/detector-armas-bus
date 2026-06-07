import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configuración de la página
st.set_page_config(page_title="Detector de Amenazas en Buses", page_icon="🚨", layout="wide")

st.title("🚨 Sistema de Detección para Buses")
st.markdown("---")

st.write("""
Esta aplicación utiliza la cámara para detectar la presencia simultánea de **motocicletas** y **personas**. 
En un entorno real, esto serviría como una alerta temprana.
""")

# Cargar el modelo YOLO
# Usamos yolov8n.pt que es el modelo más ligero y rápido
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# Clases de interés en COCO dataset (0: person, 3: motorcycle)
PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.threat_detected = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Realizar la predicción
        results = self.model(img, verbose=False)
        
        detected_person = False
        detected_motorcycle = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener la clase y coordenadas
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = ""
                color = (0, 255, 0) # Verde por defecto
                
                if class_id == PERSON_CLASS_ID and conf > 0.5:
                    detected_person = True
                    label = f"Persona {conf:.2f}"
                    color = (255, 165, 0) # Naranja
                elif class_id == MOTORCYCLE_CLASS_ID and conf > 0.5:
                    detected_motorcycle = True
                    label = f"Moto {conf:.2f}"
                    color = (0, 0, 255) # Rojo
                    
                # Dibujar bounding box
                if label:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Lógica de amenaza
        if detected_person and detected_motorcycle:
            self.threat_detected = True
            # Añadir alerta visual grande
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
            cv2.putText(img, "¡ALERTA! MOTO Y PERSONA DETECTADOS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            self.threat_detected = False
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Opciones de Prueba ---
tab1, tab2 = st.tabs(["Cámara en Vivo (WebRTC)", "Tomar Foto (Recomendado)"])

with tab1:
    st.subheader("Cámara en Vivo")
    st.write("Requiere buena conexión y puertos abiertos. Si falla, usa la pestaña 'Tomar Foto'.")
    
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:80", 
                        "turn:openrelay.metered.ca:443", 
                        "turn:openrelay.metered.ca:443?transport=tcp"
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

with tab2:
    st.subheader("Tomar Foto (Sin errores de red)")
    st.write("Toma una foto con tu celular para probar el modelo instantáneamente. ¡Funciona siempre!")
    
    foto = st.camera_input("Capturar imagen")
    
    if foto is not None:
        # Convertir la foto a un formato que OpenCV pueda leer
        bytes_data = foto.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Procesar con YOLO
        results = model(cv2_img, verbose=False)
        
        detected_person = False
        detected_motorcycle = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = ""
                color = (0, 255, 0)
                
                if class_id == PERSON_CLASS_ID and conf > 0.5:
                    detected_person = True
                    label = f"Persona {conf:.2f}"
                    color = (255, 165, 0)
                elif class_id == MOTORCYCLE_CLASS_ID and conf > 0.5:
                    detected_motorcycle = True
                    label = f"Moto {conf:.2f}"
                    color = (0, 0, 255)
                    
                if label:
                    cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        if detected_person and detected_motorcycle:
            cv2.rectangle(cv2_img, (0, 0), (cv2_img.shape[1], cv2_img.shape[0]), (0, 0, 255), 10)
            st.error("🚨 ¡ALERTA! MOTO Y PERSONA DETECTADOS 🚨")
        else:
            st.success("✅ Área segura. No se detectaron amenazas.")
            
        # Mostrar resultado
        st.image(cv2_img, channels="BGR", use_container_width=True)

# Nota sobre armas
st.markdown("---")
st.info("""
**Nota sobre detección de armas:** 
El modelo YOLOv8 estándar (COCO dataset) puede detectar personas y motos. Para detectar armas de fuego (pistolas, rifles), se requiere un modelo YOLO entrenado específicamente con un conjunto de datos de armas. 
En esta fase inicial, la alerta se dispara cuando hay personas y motocicletas simultáneamente.
""")
