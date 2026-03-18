import gradio as gr
import numpy as np
import torch
import torchvision
import cv2
from torchvision.transforms import functional as F
import math 

print("Iniciando Pytorch...")
print("Cargando el modelo keypoint R-CNN (Puede tardar unos minutos la primera vez)...")

device = torch.device('cpu')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
model.eval()

print("Modelo cargado exitosamente, cargando interfaz...")

CONEXIONES = [
    (5, 7), (7, 9),       #brazo izquierdo
    (6, 8), (8, 10),      #brazo derecho
    (5, 6), (11, 12),     #linea hombros y linea caderas
    (5, 11), (6, 12),     #torso
    (11, 13), (13, 15),   #pierna izquierda
    (12, 14), (14, 16)    #pierna derecha
]

#variables globales para el contador
contador_flexiones = 0
estado_brazo = "desconocido"

def calcular_angulo(a, b, c):
    #a hombro, b codo, c muñeca
    angulo = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if angulo < 0:
        angulo += 360
    if angulo > 180:
        angulo = 360 - angulo
    return int(angulo) #lo convertimos a entero para que no tenga decimales largos

def analizar_postura(frame):
    global contador_flexiones, estado_brazo #llamamos a las variables globales
    
    if frame is None:
        return None 
    
    imagen_dibujada = np.array(frame)
    
    tensor_img = F.to_tensor(frame).to(device)
    tensor_img = tensor_img.unsqueeze(0) 
    
    with torch.no_grad():
        predicciones = model(tensor_img)
    
    #detecta a alguien?
    if len(predicciones[0]['keypoints']) > 0:
        puntos = predicciones[0]['keypoints'][0].cpu().numpy()
        
        for punto in puntos:
            x, y, visibilidad = punto
            if visibilidad > 0.5:
                cv2.circle(imagen_dibujada, (int(x), int(y)), 5, (0, 255, 0), -1)
                
        #lineas azules para huesos
        for punto_a, punto_b in CONEXIONES:
            x_a, y_a, vis_a = puntos[punto_a]
            x_b, y_b, vis_b = puntos[punto_b]
            
            if vis_a > 0.5 and vis_b > 0.5:
                cv2.line(imagen_dibujada, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255, 0, 0), 2)

        #brazo izquierdo hombro 5, codo 7 y muñeca 9
        x5, y5, vis5 = puntos[5]
        x7, y7, vis7 = puntos[7]
        x9, y9, vis9 = puntos[9]
        
        #solo medimos si ve el brazo completo 50%confianza
        if vis5 > 0.5 and vis7 > 0.5 and vis9 > 0.5:
            angulo_brazo_izq = calcular_angulo((x5, y5), (x7, y7), (x9, y9))
            
            #escribimos el ángulo en la imagen, justo al lado del codo
            cv2.putText(imagen_dibujada, str(angulo_brazo_izq), (int(x7) + 15, int(y7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            #logica del contador
            if angulo_brazo_izq > 160:
                estado_brazo = "abajo"
            
            if angulo_brazo_izq < 50 and estado_brazo == "abajo":
                estado_brazo = "arriba"
                contador_flexiones += 1

    #interfaz visual del contador (cuadro negro bajado a Y=50)
    cv2.rectangle(imagen_dibujada, (10, 50), (250, 140), (0, 0, 0), -1)
    cv2.putText(imagen_dibujada, f"Reps: {contador_flexiones}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(imagen_dibujada, f"Estado: {estado_brazo}", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return imagen_dibujada 

interfaz = gr.Interface(
    fn=analizar_postura,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Camara en vivo"),
    outputs=gr.Image(label="Analisis Biomecanico"),
    title="KINEVISION AI",
    description="Sistema de análisis biomecánico en tiempo real.",
    live=True
)

if __name__ == "__main__":
    interfaz.launch()