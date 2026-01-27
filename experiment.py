import cv2
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

input = input.input2 # Condiciones iniciales

video_path = input['video_path']

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

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

    # Localización del ratón
    pos_real, fgmask = tracker.locate(gray)

    # Actualización de la lógica de eventos usando el centro suavizado
    logic.update(pos_real, t)

    # --- Visualización (solo para depuración) ---
    for region in regions.regions:
        state = logic.states[region.region_id]

        if state.inside:
            color = (0, 0, 255)   # rojo
        else:
            color = (0, 255, 0)   # verde

        region.draw(frame, color)
    
    # --- Visualización de la hitbox del ratón ---
    if pos_real is not None:
        inside_any = any(logic.states[r.region_id].inside for r in regions.regions)
        hitbox_color = (0, 0, 255) if inside_any else (0, 255, 0)
        x, y = pos_real
        #size = 35 #hay que cambiar para cada tamaño de ratón
        size = 10
        cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), hitbox_color, 2)

    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)

    # Salir con ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break


# --- Liberación de recursos ---
cap.release()
cv2.destroyAllWindows()
