from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
import cv2
import os
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


from .labyrinth import Labyrinth

# ==========================================================================
# Morris Pool
# ==========================================================================

class MorrisPool(Labyrinth):
	"""
	Clase para el experimento de Piscina de Morris.
	- Región de interés: un cuarto de círculo (cuadrante) donde se encuentra la plataforma.
	- Métricas: trayectoria, distancia dentro/fuera de la región, tiempo en el cuadrante
	  por evento individual y acumulado.
	"""

	def __init__(self, video_path: str, treatment: str, subject_id: str, regions: list,
				 min_detection_area: int, hitbox_size: int, start_time: float,
				 end_time: float | None = None, kernel_size=5, blur_size=0):

		super().__init__(video_path, treatment, subject_id, "MorrisPool", regions,
						 min_detection_area, hitbox_size, start_time, end_time,
						 kernel_size, blur_size)

		self._validate_morris_region()
		self.enter_frame = []
		self.exit_frame  = []
		self.event_list  = []

	# ----------------------------------------------------------------------
	# Validaciones específicas para Morris Pool
	# ----------------------------------------------------------------------

	def _validate_morris_region(self):
		# 1) Exactamente una región
		if len(self.regions.regions) != 1:
			raise ValueError(
				"Morris Pool requiere exactamente una región de interés "
				"(el cuadrante donde estuvo la plataforma)."
			)
		region = next(iter(self.regions.regions))

		# 2) Tipo correcto
		if not isinstance(region, CircularFractionRegion):
			raise ValueError("La región de Morris Pool debe ser CircularFractionRegion.")

		# 3) Verificar que sea 1/4 de círculo
		angle_span = (region.angle_end - region.angle_start) % 360
		if not np.isclose(angle_span, 90.0, atol=1e-6):
			raise ValueError("La región de Morris Pool debe ser un cuarto de círculo (90°).")

	# ----------------------------------------------------------------------
	# Procesamiento
	# ----------------------------------------------------------------------

	def process_frame(self, position, time):
		"""
		Procesa un frame a la vez para extraer la trayectoria del sujeto.

		Parámetros
		----------
		position : list
			Lista de coordenadas (x, y) detectadas en el frame actual.
		time : float
			Tiempo actual en segundos.

		Retorna
		-------
		None
		"""
		self.get_position(position, time)

	def process_video(self, all_results, all_trajectories, all_video_paths, all_first_frames, all_start_times):
		"""
		Procesa los resultados de todos los videos y genera los outputs finales.

		Parámetros
		----------
		all_results : dict
			{nombre_video: events_on_each_region}
		all_trajectories : dict
			{nombre_video: (trajectory_x, trajectory_y)}
		all_video_paths : dict
			{nombre_video: ruta_absoluta}
		all_first_frames : dict
			{nombre_video: primer_frame}
		all_start_times : dict
			{nombre_video: start_time}
		"""
		self.write_results(all_results, all_trajectories, all_video_paths, all_first_frames, all_start_times)

	# ----------------------------------------------------------------------
	# Métodos auxiliares (se mantienen por documentación histórica)
	# ----------------------------------------------------------------------

	def get_time_index_in_out_of_region(self, events_on_each_region):
		"""
		Obtiene los índices de entrada y salida de la región de interés.
		"""
		region_id        = next(iter(self.regions.regions)).region_id
		self.enter_frame = events_on_each_region[region_id].enter_frame
		self.exit_frame  = events_on_each_region[region_id].exit_frame
		self.check_enter_exit_frame_lists()

	def check_enter_exit_frame_lists(self):
		"""
		Verifica que las listas de entrada y salida tengan la misma longitud.
		Si hay una entrada sin salida, agrega una salida al final del video.
		"""
		if len(self.enter_frame) != len(self.exit_frame):
			print(f"Advertencia: entradas {len(self.enter_frame)} y salidas {len(self.exit_frame)} no coinciden.")
			if len(self.enter_frame) == len(self.exit_frame) + 1:
				print("Hay una entrada más que salidas — se agrega salida al final del video.")
				self.exit_frame.append(int(self.fps * self.end_time))
			elif len(self.enter_frame) > len(self.exit_frame):
				print("Hay más entradas que salidas. REVISAR LÓGICAS.")
			else:
				print("Hay más salidas que entradas. REVISAR LÓGICAS.")

	def get_distance_and_time_inside_region(self):
		"""
		Calcula distancia y tiempo dentro de la región para cada evento.
		"""
		for i in range(len(self.enter_frame)):
			self.event_list.append([])
			self.event_list[i].append((self.exit_frame[i] - self.enter_frame[i]) / self.fps)
			self.event_list[i].append(self.get_total_distance(
				start_frame=self.enter_frame[i], end_frame=self.exit_frame[i]))

	# ----------------------------------------------------------------------
	# Resultados
	# ----------------------------------------------------------------------

	def _write_summary(self, ws, events_on_each_region, total_distance, total_recording_time):
		"""Resumen de Morris Pool: una sola región."""
		region_id    = next(iter(self.regions.regions)).region_id
		state        = events_on_each_region[region_id]
		enter_frames = list(state.enter_frame)
		exit_frames  = list(state.exit_frame)
		enter_times  = list(state.events)  # lista de (enter_time, exit_time
		m = self._compute_region_metrics(
			enter_frames, exit_frames, enter_times,
			total_distance, total_recording_time
		)
		ws.append(["RESUMEN", ""])
		ws.append(["Nº de entradas a la región",    len(m["enter_frames"])])
		ws.append(["Tiempo total en región (s)",     round(m["total_time"], 3)])
		ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
		ws.append(["Latencia al primer ingreso (s)", round(m["latency"], 3) if m["latency"] is not None else "No entró"])
		ws.append(["% tiempo en región",             round(m["pct_time"], 2)])
		ws.append(["% distancia en región",          round(m["pct_distance"], 2)])
		ws.append([])
		self._write_event_table(ws, m)
