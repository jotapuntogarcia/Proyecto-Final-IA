import cv2
import math
import time
import os
import numpy as np
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import pygame

pygame.mixer.init()

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
if 'ultimo_audio' not in st.session_state:
    st.session_state.ultimo_audio = 0

#diccionario de explicaciones
EXPLICACIONES = {
    "MAL_SUELO": "Detecto que tu centro de masa está muy elevado. Para una flexión efectiva, debes posicionar tu cuerpo paralelo al suelo, permitiendo que la gravedad maximice la carga en tus pectorales y tríceps.",
    "MAL_CADERA_PUSH": "Atención: tu zona lumbar presenta una curvatura excesiva. Mantén el core contraído para alinear la columna; una cadera caída reduce la eficiencia y aumenta el riesgo de lesión.",
    "MAL_ESPALDA_SQUAT": "Tu torso presenta una inclinación anterior excesiva. Mantén el pecho arriba y la mirada al frente para asegurar que la carga se distribuya en los cuádriceps y no en la zona lumbar.",
    "MAL_CADERA_SQUAT": "Análisis biomecánico incompleto: el rango de movimiento es insuficiente. Para una activación glútea óptima, baja la cadera hasta que el fémur rompa la línea paralela al suelo.",
    "MAL_CODO_BICEPS": "Error de estabilidad: el codo se está desplazando del eje vertical de tu torso. Fija el codo en las costillas para aislar el bíceps y evitar compensaciones con el deltoides anterior."
}

def hablar(clave):
    ahora = time.time()
    if ahora - st.session_state.ultimo_audio > 8:
        mensaje = EXPLICACIONES.get(clave, "")
        if mensaje:
            try:
                tts = gTTS(text=mensaje, lang='es', tld='com.mx')
                tts.save("feedback.mp3")
                pygame.mixer.music.load("feedback.mp3")
                pygame.mixer.music.play()
                st.session_state.ultimo_audio = ahora
            except Exception as e:
                pass

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
        
    encender_camara = st.checkbox("Encender camara", value=False)
    
    st.markdown("---")
    ancho_video = st.slider("Ajustar tamaño del video", min_value=400, max_value=1200, value=700, step=50)
    
    st.markdown("---")
    reps_ui = st.empty()
    estado_ui = st.empty()
    postura_ui = st.empty()
    fps_ui = st.empty()

frame_window = col_video.empty()

if encender_camara:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tiempo_anterior = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        #fps
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_anterior) if tiempo_anterior != 0 else 0
        tiempo_anterior = tiempo_actual

        results = model(frame, stream=True, verbose=False, device='cpu')
        
        mensaje_postura = "Esperando postura..."
        color_postura = (255, 255, 255)

        for r in results:
            if r.keypoints is not None and r.keypoints.xy is not None and r.keypoints.conf is not None:
                keypoints = r.keypoints.xy.cpu().numpy()
                confianzas = r.keypoints.conf.cpu().numpy()
                
                if len(keypoints) > 0 and keypoints.shape[1] > 0 and len(confianzas) > 0:
                    puntos = keypoints[0]
                    confs = confianzas[0]
                    
                    if len(puntos) > 16 and len(confs) > 16:
                        visibilidad_izq = confs[5] + confs[11] + confs[13]
                        visibilidad_der = confs[6] + confs[12] + confs[14]
                        
                        if visibilidad_izq > visibilidad_der:
                            hombro = puntos[5]
                            codo = puntos[7]
                            muneca = puntos[9]
                            cadera = puntos[11]
                            rodilla = puntos[13]
                            tobillo = puntos[15]
                        else:
                            hombro = puntos[6]
                            codo = puntos[8]
                            muneca = puntos[10]
                            cadera = puntos[12]
                            rodilla = puntos[14]
                            tobillo = puntos[16]

                        if hombro[0] > 0 and codo[0] > 0 and muneca[0] > 0 and cadera[0] > 0 and tobillo[0] > 0 and rodilla[0] > 0:
                            x11, y11 = int(hombro[0]), int(hombro[1])
                            x13, y13 = int(codo[0]), int(codo[1])
                            x15, y15 = int(muneca[0]), int(muneca[1])
                            
                            #espalda/piernas
                            x_cadera, y_cadera = int(cadera[0]), int(cadera[1])
                            x_rodilla, y_rodilla = int(rodilla[0]), int(rodilla[1])
                            x_tobillo, y_tobillo = int(tobillo[0]), int(tobillo[1])

                            if st.session_state.ejercicio_actual == "Push Up":
                                angulo_brazo = calcular_angulo((x11, y11), (x13, y13), (x15, y15))
                                angulo_espalda = calcular_angulo((x11, y11), (x_cadera, y_cadera), (x_tobillo, y_tobillo))
                                
                                estas_acostado = abs(y11 - y_tobillo) < (frame.shape[0] * 0.4)
                                
                                if not estas_acostado:
                                    mensaje_postura = "MAL: PONGASE EN EL SUELO"
                                    color_postura = (0, 165, 255)
                                    hablar("MAL_SUELO")
                                elif angulo_espalda < 150:
                                    mensaje_postura = "MAL: CADERA CAIDA"
                                    color_postura = (0, 0, 255)
                                    hablar("MAL_CADERA_PUSH")
                                else:
                                    mensaje_postura = "BIEN: POSTURA RECTA"
                                    color_postura = (0, 255, 0)
                                    
                                    if angulo_brazo > 150: 
                                        st.session_state.estado_brazo = "abajo (extendido)"
                                        st.session_state['hombro_y_start'] = y11 #altura de hombro
                                        
                                    if angulo_brazo < 75 and st.session_state.estado_brazo == "abajo (extendido)":
                                        desplazamiento_y = y11 - st.session_state.get('hombro_y_start', y11)
                                        if desplazamiento_y > (frame.shape[0] * 0.1):
                                            st.session_state.estado_brazo = "arriba (flexion)"
                                            st.session_state.contador_flexiones += 1

                                cv2.putText(frame, str(angulo_brazo), (x13 + 15, y13), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.line(frame, (x11, y11), (x_cadera, y_cadera), color_postura, 2)
                                
                            elif st.session_state.ejercicio_actual == "Sentadillas":
                                angulo_pierna = calcular_angulo((x_cadera, y_cadera), (x_rodilla, y_rodilla), (x_tobillo, y_tobillo))
                                angulo_torso = calcular_angulo((x11, y11), (x_cadera, y_cadera), (x_rodilla, y_rodilla))

                                if angulo_torso < 60: #muy adelante
                                    mensaje_postura = "MAL: ESPALDA INCLINADA"
                                    color_postura = (0, 0, 255)
                                    hablar("MAL_ESPALDA_SQUAT")
                                else:
                                    mensaje_postura = "BIEN: POSTURA CORRECTA"
                                    color_postura = (0, 255, 0)
                                    
                                    if angulo_pierna > 160:
                                        st.session_state.estado_brazo = "arriba (de pie)"
                                        st.session_state['cadera_y_start'] = y_cadera 
                                        
                                    if angulo_pierna < 90 and st.session_state.estado_brazo == "arriba (de pie)":
                                        desplazamiento_y = y_cadera - st.session_state.get('cadera_y_start', y_cadera)
                                        
                                        if desplazamiento_y > (frame.shape[0] * 0.15):
                                            st.session_state.estado_brazo = "abajo (sentadilla)"
                                            st.session_state.contador_flexiones += 1
                                        else:
                                            mensaje_postura = "MAL: BAJE LA CADERA, NO SUBA LA PIERNA"
                                            color_postura = (0, 165, 255)
                                            hablar("MAL_CADERA_SQUAT")

                                cv2.putText(frame, str(angulo_pierna), (x_rodilla + 15, y_rodilla), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.line(frame, (x_cadera, y_cadera), (x_rodilla, y_rodilla), color_postura, 2)
                                cv2.line(frame, (x_rodilla, y_rodilla), (x_tobillo, y_tobillo), color_postura, 2)   
                                    
                            elif st.session_state.ejercicio_actual == "Curl de Bíceps":
                                angulo_brazo = calcular_angulo((x11, y11), (x13, y13), (x15, y15))
                                
                                #que el codo no se despegue de la cadera (trampa)
                                distancia_codo = abs(x13 - x_cadera)
                                limite_codo = frame.shape[1] * 0.12
                                
                                if distancia_codo > limite_codo: 
                                    mensaje_postura = "MAL: PEGUE EL CODO AL CUERPO"
                                    color_postura = (0, 0, 255)
                                    hablar("MAL_CODO_BICEPS")
                                else:
                                    mensaje_postura = "BIEN: POSTURA CORRECTA"
                                    color_postura = (0, 255, 0)
                                    
                                if angulo_brazo > 135:
                                    st.session_state.estado_brazo = "abajo (extendido)"
                                if angulo_brazo < 80 and st.session_state.estado_brazo == "abajo (extendido)":
                                    st.session_state.estado_brazo = "arriba (flexion)"
                                    st.session_state.contador_flexiones += 1
                                        
                                cv2.putText(frame, str(angulo_brazo), (x13 + 15, y13), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.line(frame, (x11, y11), (x13, y13), color_postura, 2)
                                cv2.line(frame, (x13, y13), (x15, y15), color_postura, 2)
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
        
        frame_window.image(frame, channels="BGR", width=ancho_video)
        
        reps_ui.metric("Repeticiones", st.session_state.contador_flexiones)
        estado_ui.info(f"Fase actual: **{st.session_state.estado_brazo}**")
        postura_ui.markdown(f"**Validación:** {mensaje_postura}")
        fps_ui.text(f"Rendimiento actual: {int(fps)} FPS")

    cap.release()
else:
    frame_window.info("Camara apagada. Ajusta los parametros y luego marca 'Encender camara'.")