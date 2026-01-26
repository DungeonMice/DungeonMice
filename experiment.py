import cv2
from tracker import MouseTracker
from regions import Region, RegionManager
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
tracker = MouseTracker()
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
    pos, fgmask = tracker.locate(gray)

    # Actualización de la lógica de eventos
    logic.update(pos, t)

    # --- Visualización (solo para depuración) ---
    for region in regions.regions:
        state = logic.states[region.id]

        if state.inside:
            color = (0, 0, 255)   # rojo
        else:
            color = (0, 255, 0)   # verde

        cv2.drawContours(frame, [region.points], -1, color, 2)

    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)

    # Salir con ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break


# --- Liberación de recursos ---
cap.release()
cv2.destroyAllWindows()
