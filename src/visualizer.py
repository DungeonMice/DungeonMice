"""Módulo de visualización para el tracking de ratones.

Se encarga de:

- Dibujar regiones de interés con colores según su estado.
- Dibujar la hitbox del ratón.
- Dibujar la trayectoria sobre la máscara de detección.
- Guardar imágenes de trayectoria al finalizar el experimento.
"""

import cv2
import numpy as np


class ExperimentVisualizer:
	"""Maneja toda la visualización del experimento.

	Attributes:
		regions: ``RegionManager`` con las regiones de interés a visualizar.
		hitbox_w: Semiancho de la hitbox del ratón en píxeles.
		hitbox_h: Semialto de la hitbox del ratón en píxeles.
	"""

	def __init__(self, regions, hitbox_size: int = 10):
		self.regions = regions
		self.hitbox_w = hitbox_size
		self.hitbox_h = hitbox_size

	def draw_regions(self, frame: np.ndarray, logic_states: dict) -> None:
		"""Dibuja todas las regiones sobre el frame con color según su estado.

		Pinta la región en rojo si el ratón está dentro y en verde si está fuera.

		Args:
			frame: Frame sobre el cual dibujar.
			logic_states: Diccionario ``{region_id: ZoneState}`` con el estado
				de cada región.
		"""
		for region in self.regions.regions:
			state = logic_states[region.region_id]
			color = (0, 0, 255) if state.inside else (0, 255, 0)
			region.draw(frame, color)

	def draw_hitbox(self, frame: np.ndarray, position, logic_states: dict) -> None:
		"""Dibuja la hitbox cuadrada alrededor de la posición del ratón.

		El color depende de si el ratón está dentro de alguna región:
		rojo si está dentro de al menos una, verde si no está en ninguna.

		Args:
			frame: Frame sobre el cual dibujar.
			position: Coordenadas (x, y) del ratón. Si es None no dibuja nada.
			logic_states: Diccionario {region_id: ZoneState} con el estado
				de cada región.
		"""
		if position is None:
			return

		inside_any = any(logic_states[r.region_id].inside for r in self.regions.regions)
		hitbox_color = (0, 0, 255) if inside_any else (0, 255, 0)

		x, y = position
		cv2.rectangle(
			frame,
			(x - self.hitbox_w, y - self.hitbox_h),
			(x + self.hitbox_w, y + self.hitbox_h),
			hitbox_color,
			2,
		)

	def draw_trajectory_on_mask(
		self,
		fgmask: np.ndarray,
		trajectory: list,
		frame_idx: int,
		draw_start_frame: int,
	) -> np.ndarray:
		"""Dibuja la trayectoria acumulada sobre la máscara de detección.

		Convierte la máscara a color y dibuja líneas conectando los puntos
		de la trayectoria grabados hasta el momento.

		Args:
			fgmask: Máscara binaria de detección en escala de grises.
			trajectory: Lista de puntos ``(x, y)`` de la trayectoria.
			frame_idx: Índice del frame actual.
			draw_start_frame: Frame a partir del cual se empieza a dibujar.

		Returns:
			Máscara con la trayectoria dibujada en color cian. Si aún no hay
			suficientes puntos, retorna la máscara convertida a BGR o en
			escala de grises según si se superó ``draw_start_frame``.
		"""
		if len(trajectory) > 1:
			fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
			for i in range(1, len(trajectory)):
				cv2.line(fgmask_color, trajectory[i - 1], trajectory[i], (255, 255, 0), 2)
			return fgmask_color
		else:
			if frame_idx >= draw_start_frame:
				return cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
			else:
				return fgmask

	def save_trajectory_image(
		self,
		video_cap,
		trajectory: list,
		total_distance: float,
		output_filename: str,
	):
		"""Guarda una imagen con la trayectoria completa sobre el primer frame.

		Args:
			video_cap: Objeto ``cv2.VideoCapture`` para leer el primer frame.
			trajectory: Lista de puntos ``(x, y)`` de la trayectoria.
			total_distance: Distancia total recorrida en píxeles.
			output_filename: Ruta del archivo de salida.

		Returns:
			Nombre del archivo guardado, o None si no se pudo generar la imagen.
		"""
		video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		ret, background = video_cap.read()

		if not ret or len(trajectory) < 2:
			return None

		# Dibujar regiones
		for region in self.regions.regions:
			region.draw(background, (0, 255, 0), 2)

		# Dibujar trayectoria completa en magenta
		for i in range(1, len(trajectory)):
			cv2.line(background, trajectory[i - 1], trajectory[i], (255, 0, 255), 2)

		# Marcar inicio (verde) y fin (rojo)
		cv2.circle(background, trajectory[0], 8, (0, 255, 0), -1)
		cv2.circle(background, trajectory[-1], 8, (0, 0, 255), -1)

		# Texto con la distancia total
		text = f"Distancia: {total_distance:.0f} px"
		cv2.putText(background, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

		cv2.imwrite(output_filename, background)
		print(f"Guardado: {output_filename}")

		return output_filename