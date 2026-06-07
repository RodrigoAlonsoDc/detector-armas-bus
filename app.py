    import subprocess
    import sys

    # --- HACK PARA STREAMLIT CLOUD ---
    # Forzamos a usar la versión "headless" (sin interfaz gráfica) de OpenCV 
    # para evitar los errores de librerías de Linux (libGL, libglib)
    try:
        import cv2
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80"])
        import cv2
    # ---------------------------------

    import streamlit as st
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

    st.subheader("Cámara de Monitoreo")
    st.write("Presiona 'START' y permite el acceso a tu cámara para iniciar la detección.")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    # Nota sobre armas
    st.markdown("---")
    st.info("""
    **Nota sobre detección de armas:** 
    El modelo YOLOv8 estándar (COCO dataset) puede detectar personas y motos. Para detectar armas de fuego (pistolas, rifles), se requiere un modelo YOLO entrenado específicamente con un conjunto de datos de armas. 
    En esta fase inicial, la alerta se dispara cuando hay personas y motocicletas simultáneamente.
    """)
    
