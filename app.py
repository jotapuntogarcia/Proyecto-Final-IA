import gradio as gr
import numpy as np
import torch
import torchvision
import cv2 #biuja los puntos
from torchvision.transforms import functional as F  

print("Iniciando Pytorch...")
print("Cargando el modelo keypoint R-CNN (Puede tardar unos minutos la primera vez)...")

device = torch.device('cpu')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
model.eval()

print("Modelo cargado exitosamente, cargando interfaz...")

def analizar_postura(frame):
    if frame is None:
        return None 
    
    #copia para dibujar los puntos
    imagen_dibujada = np.array(frame)
    
    tensor_img = F.to_tensor(frame).to(device)
    tensor_img = tensor_img.unsqueeze(0) 
    
    with torch.no_grad():
        predicciones = model(tensor_img)
    
    #si detecta a alguien, dibujamos los 17 puntos
    if len(predicciones[0]['keypoints']) > 0:
        puntos = predicciones[0]['keypoints'][0].cpu().numpy()
        
        for punto in puntos:
            x, y, visibilidad = punto
            #dibujamos si la confianza es mayor al 50%
            if visibilidad > 0.5:
                cv2.circle(imagen_dibujada, (int(x), int(y)), 5, (0, 255, 0), -1)

    return imagen_dibujada 

interfaz = gr.Interface(
    fn=analizar_postura,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="camara en vivo"),
    outputs=gr.Image(label="Analisis Biomecanico"),
    title="Coach",
    description="Paso 2: IA detectando articulaciones en tiempo real",
    live=True
)

if __name__ == "__main__":
    interfaz.launch()