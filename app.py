import gradio as gr
import numpy as np
import cv2
import math 

#importación blindada para el motor de mediapipe
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

print("Iniciando KINEVISION AI con Motor MediaPipe...")

#configuramos el motor de pose de mediapipe
pose_engine = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

#variables globales para el contador
contador_flexiones = 0
estado_brazo = "desconocido"

def calcular_angulo(a, b, c):
    # a hombro, b codo, c muñeca
    angulo = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if angulo < 0:
        angulo += 360
    if angulo > 180:
        angulo = 360 - angulo
    return int(angulo) 

def analizar_postura(frame):
    global contador_flexiones, estado_brazo 
    
    if frame is None:
        return None 
    
    imagen_dibujada = np.array(frame)

    #procesamiento con mediapipe en rgb
    imagen_rgb = cv2.cvtColor(imagen_dibujada, cv2.COLOR_BGR2RGB)
    resultados = pose_engine.process(imagen_rgb)
    
    if resultados.pose_landmarks:
        #dibujamos el esqueleto y los puntos directamente en la imagen
        mp_drawing.draw_landmarks(imagen_dibujada, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        puntos = resultados.pose_landmarks.landmark
        alto, ancho, _ = imagen_dibujada.shape
        
        #puntos del brazo izquierdo en mediapipe: hombro(11), codo(13), muñeca(15)
        hombro_izq = puntos[11]
        codo_izq = puntos[13]
        muneca_izq = puntos[15]
        
        #solo medimos y mostramos si la IA ve claramente el brazo
        if hombro_izq.visibility > 0.5 and codo_izq.visibility > 0.5 and muneca_izq.visibility > 0.5:
            #pasamos las coordenadas normalizadas a píxeles reales
            x11, y11 = int(hombro_izq.x * ancho), int(hombro_izq.y * alto)
            x13, y13 = int(codo_izq.x * ancho), int(codo_izq.y * alto)
            x15, y15 = int(muneca_izq.x * ancho), int(muneca_izq.y * alto)

            angulo_brazo_izq = calcular_angulo((x11, y11), (x13, y13), (x15, y15))
            
            #escribimos el ángulo en la imagen, justo al lado del codo
            cv2.putText(imagen_dibujada, str(angulo_brazo_izq), (x13 + 15, y13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            #logica del contador con los angulos ajustados para mayor fluidez
            if angulo_brazo_izq > 150: 
                estado_brazo = "abajo (extendido)"
            
            if angulo_brazo_izq < 75 and estado_brazo == "abajo (extendido)": 
                estado_brazo = "arriba (flexion)"
                contador_flexiones += 1

    #interfaz visual del cuadro negro
    cv2.rectangle(imagen_dibujada, (10, 50), (320, 160), (0, 0, 0), -1)
    cv2.putText(imagen_dibujada, f"Reps: {contador_flexiones}", (25, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(imagen_dibujada, f"Estado: {estado_brazo}", (25, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return imagen_dibujada 

interfaz = gr.Interface(
    fn=analizar_postura,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Camara en navegador"),
    outputs=gr.Image(label="Analisis KineVision"),
    title="KINEVISION AI",
    description="Sistema de análisis biomecánico en tiempo real.",
    live=True
)

if __name__ == "__main__":
    interfaz.launch()