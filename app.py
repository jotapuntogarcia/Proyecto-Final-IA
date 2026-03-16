import gradio as gr
import numpy as hp

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
    