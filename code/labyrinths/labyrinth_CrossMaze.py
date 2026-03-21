from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
import cv2
import os
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


from .labyrinth import Labyrinth

# ==========================================================================
# Cross Maze
# ==========================================================================

class CrossMaze(Labyrinth):
	"""
	Clase para laberintos en cruz.
	- Soporta cualquier número de regiones poligonales (brazos).
	- Métricas por región: latencia, entradas, tiempo, distancia, porcentajes.
	"""

	def __init__(self, video_path, treatment, subject_id, regions,
				 min_detection_area, hitbox_size, start_time, end_time=None,
				 kernel_size=5, blur_size=0):

		super().__init__(video_path, treatment, subject_id, "CrossMaze", regions,
						 min_detection_area, hitbox_size, start_time, end_time,
						 kernel_size, blur_size)

		self._validate_cross_maze_regions()

	# ----------------------------------------------------------------------
	# Validaciones específicas para Cross Maze
	# ----------------------------------------------------------------------

	def _validate_cross_maze_regions(self):
		if len(self.regions.regions) < 2:
			raise ValueError("CrossMaze requiere al menos 2 regiones de interés.")
		for region in self.regions.regions:
			if not isinstance(region, PolygonRegion):
				raise ValueError("Todas las regiones de CrossMaze deben ser PolygonRegion.")

	# ----------------------------------------------------------------------
	# Procesamiento
	# ----------------------------------------------------------------------

	def process_frame(self, position, time):
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
		"""
		self.write_results(all_results, all_trajectories, all_video_paths, all_first_frames, all_start_times)

	# ----------------------------------------------------------------------
	# Resultados
	# ----------------------------------------------------------------------

	def _write_summary(self, ws, events_on_each_region, total_distance, total_recording_time):
		"""Resumen de CrossMaze: una sección por región."""
		ws.append(["RESUMEN GLOBAL", ""])
		ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
		ws.append(["Duración total grabación (s)",   round(len(self.trajectory_x) / self.fps, 3)])
		ws.append([])

		for region in self.regions.regions:
			state = events_on_each_region[region.region_id]
			enter_frames = list(state.enter_frame)
			exit_frames  = list(state.exit_frame)
			enter_times  = list(state.events)  # lista de (enter_time, exit_time)
			m = self._compute_region_metrics(
						enter_frames, exit_frames, enter_times,
						total_distance, len(self.trajectory_x) / self.fps
					)
			ws.append([f"REGIÓN: {region.region_id}", ""])
			ws.append(["Nº de entradas",                len(m["enter_frames"])])
			ws.append(["Tiempo total en región (s)",     round(m["total_time"], 3)])
			ws.append(["Distancia en región (px)",       round(m["total_dist"], 2)])
			ws.append(["Latencia al primer ingreso (s)", round(m["latency"], 3) if m["latency"] is not None else "No entró"])
			ws.append(["% tiempo en región",             round(m["pct_time"], 2)])
			ws.append(["% distancia en región",          round(m["pct_distance"], 2)])
			ws.append([])
			self._write_event_table(ws, m)
			ws.append([])  # separador entre regiones