import gradio as gr
import numpy as hp
import torch
import torchvision

print("Iniciando Pytorch...")
print("Cargando el modelo keypoint R-CNN (Puede tardar unos minutos la primera vez)...")

device = torch.device('cpu')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
model.eval()

print("Modelo cargado exitosamente, cargando interfaz...")

def analizar_postura(frame):
    return frame 

interfaz = gr.Interface(
    fn=analizar_postura,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="camara en vivo"),
    outputs=gr.Image(label="Analisis Biomecanico"),
    title="Coach",
    description="Paso 1: Verificando conexion de camara"
)

if __name__ == "__main__":
    interfaz.launch()
    