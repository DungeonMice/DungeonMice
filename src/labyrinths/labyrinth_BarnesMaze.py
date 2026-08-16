"""Implementación del laberinto de Barnes (Barnes Maze)."""

from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
import cv2
import os
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from .labyrinth import Labyrinth

class BarnesMaze(Labyrinth):
	"""Laberinto Barnes circular con n número de regiones circulares (Agujeros).
	
	Métricas calculadas por región: latencia para encontrar el agujero correcto 
	(tiempo), tiempo en el 'cuadrante' objetivo, número de errores.
	
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
			mog_threshold: int = 30,
			recording_lr: float = 0.002,
			use_csrt: bool = True,
			mog_history: int = 500,
	):
		"""
		Inicializa el laberinto Barnes circular.

		Attributes:
			video_path: Ruta al video o carpeta de videos.
			treatment: Nombre del tratamiento.
			subject_id: ID del sujeto.
			regions: Regiones de interés (todas CircleRegion, mínimo 2).
			min_detection_area: Área mínima del blob en píxeles.
			hitbox_size: Semilado de la hitbox en píxeles.
			start_time: Segundo de inicio del registro.
			end_time: Segundo de fin, o None.
			kernel_size: Kernel morfológico del tracker.
			blur_size: GaussianBlur. 0 = sin blur.
			mog_threshold: Umbral de varianza MOG2.
			recording_lr: Learning rate MOG2 durante grabación.
			use_csrt: Activar CSRT como tracker primario.
			mog_history: Frames para el modelo de fondo de MOG2.

		Raises:
			ValueError: Si hay menos de 2 regiones o alguna no es CircleRegion.
		"""

		super().__init__(
					video_path, treatment, subject_id, "BarnesMaze", regions,
					min_detection_area, hitbox_size, start_time, end_time,
					kernel_size, blur_size, mog_threshold, recording_lr,
					use_csrt, mog_history,
				)

		# Validaciones de las regiones para el laberinto Barnes
		self.validate_barnes_maze_regions()


	# ------------------------------------------------------------------
	# Validaciones específicas
	# ------------------------------------------------------------------
	
	def validate_barnes_maze_regions(self) -> None:
		"""Valida que las regiones sean compatibles con BarnesMaze.

		Raises:
			ValueError: Si hay menos de 2 regiones o alguna no es CircleRegion.
		"""
		if len(self.regions.regions) < 2:
			raise ValueError("Se requieren al menos 2 regiones para BarnesMaze.")
		for region in self.regions.regions:
			if not isinstance(region, CircleRegion):
				raise ValueError(
					f"Todas las regiones deben ser CircleRegion para BarnesMaze. "
					f"Se encontró {type(region).__name__}."
				)

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
		all_recording_durations: dict,
		) -> None:
			
		"""Genera los outputs finales para todos los videos procesados.
	
		Args:
			all_results: ``{nombre_video: events_on_each_region}``.
			all_trajectories: ``{nombre_video: (trajectory_x, trajectory_y)}``.
			all_video_paths: ``{nombre_video: ruta_absoluta}``.
			all_first_frames: ``{nombre_video: primer_frame}``.
			all_start_times: ``{nombre_video: start_time}``.
			all_recording_durations: ``{nombre_video: duracion_grabacion_segundos}``
				con la duración real grabada por video.
		"""
		self.write_results(
			all_results, all_trajectories, all_video_paths,
			all_first_frames, all_start_times, all_recording_durations,
			)

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
			"""Escribe el resumen de BarnesMaze en la hoja de Excel.
	
			Incluye las métricas globales de la única región de interés y la
			tabla detallada de eventos de entrada/salida.
	
			Args:
				ws: Hoja de Excel (``Worksheet``) donde escribir.
				events_on_each_region: ``{region_id: ZoneState}`` con los eventos
					de la región.
				total_distance: Distancia total recorrida en píxeles.
				total_recording_time: Duración total de la grabación en segundos.
			"""
			region_id    = next(iter(self.regions.regions)).region_id
			state        = events_on_each_region[region_id]
			enter_frames = list(state.enter_frame)
			exit_frames  = list(state.exit_frame)
			enter_times  = list(state.events)
			m = self.compute_region_metrics(
				enter_frames, exit_frames, enter_times,
				total_distance, total_recording_time,
			)
			ws.append(["RESUMEN", ""])
			ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
			ws.append(["Latencia al primer ingreso (s)", round(m["latency"], 3) if m["latency"] is not None else "No entró"])
			#ws.append(["% tiempo en región",             round(m["pct_time"], 2)])
			#ws.append(["% distancia en región",          round(m["pct_distance"], 2)])
			ws.append([])
			self.write_event_table(ws, m)