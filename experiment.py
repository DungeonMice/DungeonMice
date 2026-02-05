import cv2
import numpy as np
from tracker import MouseTracker
from logic import EventLogic
import input

"""
experiment.py

Script principal de ejecución del experimento.

Este archivo se encarga de:
- Cargar el video
- Iterar frame por frame
- Calcular el tiempo real asociado a cada frame
- Coordinar la detección del ratón, la evaluación de regiones
  y la lógica de eventos

No contiene lógica de negocio reutilizable. Su función es
integrar componentes que ya funcionan de forma independiente.
"""

# --- Inicialización del video ---

input = input.input3 # Condiciones iniciales

video_path = input['video_path']

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

# Frame a partir del cual se empieza a DIBUJAR
draw_start_frame = int(fps*6)  # *numero segundos después

# --- Definición de regiones de interés ---
# Más adelante podrán venir de mouse, archivo o GUI,
# sin cambiar el resto del backend.
regions = input['regions']

# --- Inicialización de módulos del backend ---
tracker = MouseTracker(min_area=100) #poner min_area=100 para otro video
logic = EventLogic(regions)

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

    # Actualización de la lógica de eventos usando el centro suavizado
    logic.update(pos_real, t)
    
    # Dibujar trayectoria en la máscara si ya hay puntos grabados
    if len(tracker.trajectory) > 1:
        fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        
        for i in range(1, len(tracker.trajectory)):
            pt1 = tracker.trajectory[i-1]
            pt2 = tracker.trajectory[i]
            cv2.line(fgmask_color, pt1, pt2, (255, 255, 0), 2)
        
        fgmask = fgmask_color
    else:
        # Convertir máscara a color después del delay para mantener consistencia
        if frame_idx >= draw_start_frame:
            fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)


    # Visualización de regiones de interés
    for region in regions.regions:
        state = logic.states[region.region_id]

        if state.inside:
            color = (0, 0, 255)   # rojo
        else:
            color = (0, 255, 0)   # verde

        region.draw(frame, color)
    
    # Visualización de la hitbox del ratón 
    if pos_real is not None:
        inside_any = any(logic.states[r.region_id].inside for r in regions.regions)
        hitbox_color = (0, 0, 255) if inside_any else (0, 255, 0)
        x, y = pos_real
        #size = 30 #hay que cambiar para cada tamaño de video
        size = 10
        cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), hitbox_color, 2)

    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)

    # Salir con ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break
    
# Calcular distancia total recorrida
total_distance = tracker.get_total_distance()
print(f"Distancia total: {total_distance:.2f} pixeles")

# --- Guardar imagen de trayectoria sobre el primer frame ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, background = cap.read()

if ret and len(tracker.trajectory) > 1:
    # Dibujar regiones
    for region in regions.regions:
        region.draw(background, (0, 255, 0), 2)
    
    # Dibujar trayectoria completa
    for i in range(1, len(tracker.trajectory)):
        pt1 = tracker.trajectory[i-1]
        pt2 = tracker.trajectory[i]
        cv2.line(background, pt1, pt2, (255, 0, 255), 2)
    
    # Marcar inicio (verde) y fin (rojo)
    cv2.circle(background, tracker.trajectory[0], 8, (0, 255, 0), -1)
    cv2.circle(background, tracker.trajectory[-1], 8, (0, 0, 255), -1)
    
    # ← Agregar texto con la distancia total
    text = f"Distancia: {total_distance:.0f} px"
    cv2.putText(background, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Guardar imagen
    video_name = input['video_path'].split('.')[0]
    cv2.imwrite(f"trajectory_{video_name}.png", background)
    print(f"Guardado: trajectory_{video_name}.png")

# --- Liberación de recursos ---
cap.release()
cv2.destroyAllWindows()
