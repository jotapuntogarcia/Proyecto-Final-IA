import gradio as gr
import numpy as np
import torch
import torchvision
import cv2
from torchvision.transforms import functional as F  

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

def analizar_postura(frame):
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

    return imagen_dibujada 

interfaz = gr.Interface(
    fn=analizar_postura,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Camara en vivo"),
    outputs=gr.Image(label="Analisis Biomecanico"),
    title="Coach",
    description="Paso 3: Esqueleto biomecánico conectado",
    live=True
)

if __name__ == "__main__":
    interfaz.launch()