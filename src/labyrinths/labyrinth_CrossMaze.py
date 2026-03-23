"""Implementación del laberinto en cruz (Cross Maze)."""

from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
import cv2
import os
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from .labyrinth import Labyrinth


class CrossMaze(Labyrinth):
	"""Laberinto en cruz con cualquier número de regiones poligonales (brazos).

	Métricas calculadas por región: latencia al primer ingreso, número de
	entradas, tiempo acumulado, distancia recorrida y porcentajes respecto
	al total de la grabación.
	"""

	def __init__(
		self,
		video_path: str,
		treatment: str,
		subject_id: str,
		regions: RegionManager,
		min_detection_area: int,
		hitbox_size: int,
		start_time: float,
		end_time: float | None = None,
		kernel_size: int = 5,
		blur_size: int = 0,
	):
		super().__init__(
			video_path, treatment, subject_id, "CrossMaze", regions,
			min_detection_area, hitbox_size, start_time, end_time,
			kernel_size, blur_size,
		)
		self.validate_cross_maze_regions()

	# ------------------------------------------------------------------
	# Validaciones específicas
	# ------------------------------------------------------------------

	def validate_cross_maze_regions(self) -> None:
		"""Valida que las regiones sean compatibles con CrossMaze.

		Raises:
			ValueError: Si hay menos de 2 regiones o alguna no es PolygonRegion.
		"""
		if len(self.regions.regions) < 2:
			raise ValueError("CrossMaze requiere al menos 2 regiones de interes.")
		for region in self.regions.regions:
			if not isinstance(region, PolygonRegion):
				raise ValueError("Todas las regiones de CrossMaze deben ser PolygonRegion.")

	# ------------------------------------------------------------------
	# Procesamiento
	# ------------------------------------------------------------------

	def process_frame(self, position: tuple | None, time: float) -> None:
		"""Almacena la posición del frame actual en la trayectoria.

		Args:
			position: Coordenadas (x, y) del ratón detectado, o None si no
				se detectó.
			time: Tiempo actual en segundos.
		"""
		self.get_position(position, time)

	def process_video(
		self,
		all_results: dict,
		all_trajectories: dict,
		all_video_paths: dict,
		all_first_frames: dict,
		all_start_times: dict,
	) -> None:
		"""Genera los outputs finales para todos los videos procesados.

		Args:
			all_results: {nombre_video: events_on_each_region}.
			all_trajectories: {nombre_video: (trajectory_x, trajectory_y)}.
			all_video_paths: {nombre_video: ruta_absoluta}.
			all_first_frames: {nombre_video: primer_frame}.
			all_start_times: {nombre_video: start_time}.
		"""
		self.write_results(all_results, all_trajectories, all_video_paths, all_first_frames, all_start_times)

	# ------------------------------------------------------------------
	# Resultados
	# ------------------------------------------------------------------

	def write_summary(
		self,
		ws,
		events_on_each_region: dict,
		total_distance: float,
		total_recording_time: float,
	) -> None:
		"""Escribe el resumen de CrossMaze en la hoja de Excel.

		Genera una sección de resumen global con métricas por tipo de brazo
		(abierto: este-oeste, cerrado: norte-sur) y una sección detallada
		por cada región con su tabla de eventos de entrada/salida.

		Args:
			ws: Hoja de Excel (Worksheet) donde escribir.
			events_on_each_region: {region_id: ZoneState} con los eventos
				de cada región.
			total_distance: Distancia total recorrida en píxeles.
			total_recording_time: Duración total de la grabación en segundos.
		"""
		recording_time = len(self.trajectory_x) / self.fps

		# Calcular métricas por región
		metrics = {}
		for region in self.regions.regions:
			state        = events_on_each_region[region.region_id]
			enter_frames = list(state.enter_frame)
			exit_frames  = list(state.exit_frame)
			enter_times  = list(state.events)
			metrics[region.region_id] = self.compute_region_metrics(
				enter_frames, exit_frames, enter_times,
				total_distance, recording_time,
			)

		# Agrupar por tipo de brazo
		open_ids   = [r.region_id for r in self.regions.regions if r.region_id.lower() in ("este", "oeste")]
		closed_ids = [r.region_id for r in self.regions.regions if r.region_id.lower() in ("norte", "sur")]

		def arm_totals(ids):
			entries  = sum(len(metrics[i]["enter_frames"]) for i in ids if i in metrics)
			time     = sum(metrics[i]["total_time"]        for i in ids if i in metrics)
			distance = sum(metrics[i]["total_dist"]        for i in ids if i in metrics)
			pct_time = (time     / recording_time * 100) if recording_time > 0 else 0
			pct_dist = (distance / total_distance * 100) if total_distance > 0 else 0
			return entries, time, distance, pct_time, pct_dist

		open_entries,   open_time,   open_dist,   open_pct_time,   open_pct_dist   = arm_totals(open_ids)
		closed_entries, closed_time, closed_dist, closed_pct_time, closed_pct_dist = arm_totals(closed_ids)

		# --- Resumen global ---
		ws.append(["RESUMEN GLOBAL", ""])
		ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
		ws.append(["Duración total grabación (s)",   round(recording_time, 3)])
		ws.append([])

		ws.append(["BRAZOS ABIERTOS (este + oeste)", ""])
		ws.append(["Nº de entradas",        open_entries])
		ws.append(["Tiempo total (s)",      round(open_time, 3)])
		ws.append(["Distancia total (px)",  round(open_dist, 2)])
		ws.append(["% tiempo",              round(open_pct_time, 2)])
		ws.append(["% distancia",           round(open_pct_dist, 2)])
		ws.append([])

		ws.append(["BRAZOS CERRADOS (norte + sur)", ""])
		ws.append(["Nº de entradas",        closed_entries])
		ws.append(["Tiempo total (s)",      round(closed_time, 3)])
		ws.append(["Distancia total (px)",  round(closed_dist, 2)])
		ws.append(["% tiempo",              round(closed_pct_time, 2)])
		ws.append(["% distancia",           round(closed_pct_dist, 2)])
		ws.append([])

		# --- Detalle por región ---
		for region in self.regions.regions:
			m = metrics[region.region_id]
			ws.append([f"REGIÓN: {region.region_id}", ""])
			ws.append(["Nº de entradas",                len(m["enter_frames"])])
			ws.append(["Tiempo total en región (s)",     round(m["total_time"], 3)])
			ws.append(["Distancia en región (px)",       round(m["total_dist"], 2)])
			ws.append(["Latencia al primer ingreso (s)", round(m["latency"], 3) if m["latency"] is not None else "No entró"])
			ws.append(["% tiempo en región",             round(m["pct_time"], 2)])
			ws.append(["% distancia en región",          round(m["pct_distance"], 2)])
			ws.append([])
			self.write_event_table(ws, m)
			ws.append([])