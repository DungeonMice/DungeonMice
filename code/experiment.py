import cv2
import numpy as np
from tracker import MouseTracker
from logic import EventLogic
from visualizer import ExperimentVisualizer
import input

"""
experiment.py

Script principal de ejecución del experimento.

Este archivo se encarga de:
- Cargar el video
- Iterar frame por frame
- Calcular el tiempo real asociado a cada frame
- Coordinar la detección del ratón, la evaluación de regiones y la lógica de eventos
"""

# --- Inicialización del video ---

input = input.input3 # Condiciones iniciales
# Poner que es input1, input2 o input3 para cada video.
# Hay que cambiar también el min_area del tracker para cada video.

video_path = input['video_path']

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

# Frame a partir del cual se empieza a DIBUJAR
draw_start_frame = int(fps*6)  # *numero segundos después
# Esto habría que ponerlo en el input también, para cada video, dependiendo de cuándo empieza a moverse el ratón.

# --- Definición de regiones de interés ---
# Más adelante podrán venir de seleccionar con el mouse,
# sin cambiar el resto del backend.
regions = input['regions']

# --- Inicialización de módulos del backend ---
tracker = MouseTracker(min_area=100) # Hay que ajustar el min_area para cada video, dependiendo del tamaño.
logic = EventLogic(regions)
visualizer = ExperimentVisualizer(regions, hitbox_size=10)

# --- Loop principal del experimento ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tiempo actual en segundos
    t = frame_idx / fps
    frame_idx += 1

    # Conversión a escala de grises para el detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Activar grabación de trayectoria cuando se alcance el frame indicado
    if frame_idx == draw_start_frame:
        tracker.start_recording()

    # Localización del ratón
    pos_real, fgmask = tracker.locate(gray)

    # Actualización de la lógica de eventos
    logic.update(pos_real, t)
    
    # --- Visualización ---
    fgmask = visualizer.draw_trajectory_on_mask(fgmask, tracker.trajectory, 
                                                frame_idx, draw_start_frame)
    visualizer.draw_regions(frame, logic.states)
    visualizer.draw_hitbox(frame, pos_real, logic.states)
    
    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)

    # Salir con ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break
    
# Calcular distancia total recorrida
total_distance = tracker.get_total_distance()
print(f"Distancia total: {total_distance:.2f} pixeles")

# Guardar imagen de la trayectoria
video_name = input['video_path'].split('.')[0]
visualizer.save_trajectory_image(cap, tracker.trajectory, total_distance, 
                                f"trajectory_{video_name}.png")
# --- Liberación de recursos ---
cap.release()
cv2.destroyAllWindows()
