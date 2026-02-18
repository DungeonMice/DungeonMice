import cv2
import numpy as np
from tracker import MouseTracker
from logic import EventLogic
from visualizer import ExperimentVisualizer
from input import input1, input2, input3


def main(labyrinth):

	# --- Abrir video ---
	cap = cv2.VideoCapture(labyrinth.video_path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	
	labyrinth.fps = fps # Asignar el numero de fps

	frame_idx = 0

	tracker = MouseTracker(min_area=labyrinth.min_detection_area)
	logic = EventLogic(labyrinth.regions)
	visualizer = ExperimentVisualizer(labyrinth.regions, hitbox_size=labyrinth.hitbox_size)

	# --- Loop principal ---
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		t = frame_idx / fps
		frame_idx += 1

		# Conversión a escala de grises para el detector
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Activar grabación de trayectoria cuando se alcance el frame indicado
		if frame_idx == labyrinth.start_time * fps:
			tracker.start_recording()

		# Localizar la posición del ratón
		pos_real, fgmask = tracker.locate(gray)
		trajectory = tracker.trajectory

		# Actualizar la lógica de eventos
		logic.update(pos_real, t, frame_idx)

		# Visualización
		fgmask = visualizer.draw_trajectory_on_mask(fgmask, tracker.trajectory, frame_idx, labyrinth.start_time * fps)
		visualizer.draw_regions(frame, logic.states)
		visualizer.draw_hitbox(frame, pos_real, logic.states)

		# Procesamiento del frame
		labyrinth.process_frame(position = trajectory ,time = t)
  
		# Mostrar timestamp en el frame
		cv2.putText(
			frame,
			f"t = {t:.2f} s",
			(10, 20),  # posición en el frame
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,       # tamaño de fuente, ajusta si se ve muy grande/pequeño
			(255, 255, 255),  # color blanco
			1
		)

		cv2.imshow("frame", frame)
		cv2.imshow("fgmask", fgmask)

		# Salir con ESC
		if cv2.waitKey(30) & 0xFF == 27:
			break

	# --- Finalización ---
	labyrinth.process_video(events_on_each_region = logic.states)

	cap.release()
	cv2.destroyAllWindows()
