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
if 'ejercicio_actual' not in st.session_state:
    st.session_state.ejercicio_actual = "Push Up"

def calcular_angulo(a, b, c):
    angulo = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if angulo < 0: angulo += 360
    if angulo > 180: angulo = 360 - angulo
    return int(angulo)

col_video, col_controles = st.columns([3, 1])

with col_controles:
    st.write("### Panel de Control")
    
    ejercicio_seleccionado = st.selectbox("Selecciona Ejercicio", ["Push Up", "Sentadillas", "Curl de Bíceps"])
    
    if st.session_state.ejercicio_actual != ejercicio_seleccionado:
        st.session_state.contador_flexiones = 0
        st.session_state.estado_brazo = "desconocido"
        st.session_state.ejercicio_actual = ejercicio_seleccionado

    if st.button("Resetear Contador", type="primary"):
        st.session_state.contador_flexiones = 0
    stop_button = st.button("Detener Cámara")
    
    st.markdown("---")
    reps_ui = st.empty()
    estado_ui = st.empty()
    postura_ui = st.empty()
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
    
    mensaje_postura = "Esperando postura..."
    color_postura = (255, 255, 255)

    for r in results:
        if r.keypoints is not None and r.keypoints.xy is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            
            if len(keypoints) > 0 and keypoints.shape[1] > 0:
                puntos = keypoints[0]
                
                if len(puntos) > 15:
                    hombro = puntos[5]
                    codo = puntos[7]
                    muneca = puntos[9]
                    cadera = puntos[11]
                    tobillo = puntos[15]

                    if hombro[0] > 0 and codo[0] > 0 and muneca[0] > 0 and cadera[0] > 0 and tobillo[0] > 0:
                        x11, y11 = int(hombro[0]), int(hombro[1])
                        x13, y13 = int(codo[0]), int(codo[1])
                        x15, y15 = int(muneca[0]), int(muneca[1])
                        
                        #espalda
                        x_cadera, y_cadera = int(cadera[0]), int(cadera[1])
                        x_tobillo, y_tobillo = int(tobillo[0]), int(tobillo[1])

                        if st.session_state.ejercicio_actual == "Push Up":
                            angulo_brazo_izq = calcular_angulo((x11, y11), (x13, y13), (x15, y15))
                            angulo_espalda = calcular_angulo((x11, y11), (x_cadera, y_cadera), (x_tobillo, y_tobillo))
                            
                            if angulo_espalda < 150:
                                mensaje_postura = "MAL: CADERA CAIDA"
                                color_postura = (0, 0, 255)
                            else:
                                mensaje_postura = "BIEN: POSTURA RECTA"
                                color_postura = (0, 255, 0)
                                
                                if angulo_brazo_izq > 150: 
                                    st.session_state.estado_brazo = "abajo (extendido)"
                                if angulo_brazo_izq < 75 and st.session_state.estado_brazo == "abajo (extendido)":
                                    st.session_state.estado_brazo = "arriba (flexion)"
                                    st.session_state.contador_flexiones += 1

                            cv2.putText(frame, str(angulo_brazo_izq), (x13 + 15, y13), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.line(frame, (x11, y11), (x_cadera, y_cadera), color_postura, 2)
            else:
                mensaje_postura = "Usuario no detectado"
                color_postura = (150, 150, 150)

    cv2.rectangle(frame, (10, 50), (400, 230), (0, 0, 0), -1) 
    cv2.putText(frame, f"Reps: {st.session_state.contador_flexiones}", (25, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Estado: {st.session_state.estado_brazo}", (25, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Tecnica: {mensaje_postura}", (25, 175), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_postura, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (25, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_window.image(frame, channels="BGR")
    
    reps_ui.metric("Repeticiones", st.session_state.contador_flexiones)
    estado_ui.info(f"Fase actual: **{st.session_state.estado_brazo}**")
    postura_ui.markdown(f"**Validación:** {mensaje_postura}")
    fps_ui.text(f"Rendimiento actual: {int(fps)} FPS")

cap.release()