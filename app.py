import cv2
import math
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

#configuración interfaz
st.set_page_config(page_title="KINEVISION AI", layout="wide")
st.title("KINEVISION AI")
st.markdown("Sistema de análisis biomecánico en tiempo real")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt') 

model = load_model()

#variables
if 'contador_flexiones' not in st.session_state:
    st.session_state.contador_flexiones = 0
if 'estado_brazo' not in st.session_state:
    st.session_state.estado_brazo = "desconocido"

def calcular_angulo(a, b, c):
    angulo = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if angulo < 0: angulo += 360
    if angulo > 180: angulo = 360 - angulo
    return int(angulo)

col_video, col_controles = st.columns([3, 1])

with col_controles:
    st.write("### Panel de Control")
    if st.button("Resetear Contador", type="primary"):
        st.session_state.contador_flexiones = 0
    stop_button = st.button("Detener Cámara")
    
    st.markdown("---")
    reps_ui = st.empty()
    estado_ui = st.empty()
    fps_ui = st.empty()

frame_window = col_video.empty()

#camara
cap = cv2.VideoCapture(0)
tiempo_anterior = 0

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret: break
    
    #fps
    tiempo_actual = time.time()
    fps = 1 / (tiempo_actual - tiempo_anterior) if tiempo_anterior != 0 else 0
    tiempo_anterior = tiempo_actual

    results = model(frame, stream=True, verbose=False, device='cpu')
    
    for r in results:
        if r.keypoints is not None and r.keypoints.xy is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            
            if keypoints.shape[1] > 0:
                puntos = keypoints[0]
                
                #izqquierda 
                if len(puntos) > 9:
                    hombro = puntos[5]
                    codo = puntos[7]
                    muneca = puntos[9]

                    if hombro[0] > 0 and codo[0] > 0 and muneca[0] > 0:
                        x11, y11 = int(hombro[0]), int(hombro[1])
                        x13, y13 = int(codo[0]), int(codo[1])
                        x15, y15 = int(muneca[0]), int(muneca[1])

                        angulo_brazo_izq = calcular_angulo((x11, y11), (x13, y13), (x15, y15))
                        
                        if angulo_brazo_izq > 150: 
                            st.session_state.estado_brazo = "abajo (extendido)"
                        if angulo_brazo_izq < 75 and st.session_state.estado_brazo == "abajo (extendido)":
                            st.session_state.estado_brazo = "arriba (flexion)"
                            st.session_state.contador_flexiones += 1

                        #angulo linea codo
                        cv2.putText(frame, str(angulo_brazo_izq), (x13 + 15, y13), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (10, 50), (320, 190), (0, 0, 0), -1) 
    cv2.putText(frame, f"Reps: {st.session_state.contador_flexiones}", (25, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Estado: {st.session_state.estado_brazo}", (25, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (25, 175), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_window.image(frame, channels="BGR")
    
    reps_ui.metric("Flexiones Completadas", st.session_state.contador_flexiones)
    estado_ui.info(f"Fase actual: **{st.session_state.estado_brazo}**")
    fps_ui.text(f"Rendimiento actual: {int(fps)} FPS")

cap.release()