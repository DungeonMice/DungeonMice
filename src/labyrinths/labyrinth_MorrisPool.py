"""Implementación del experimento de Piscina de Morris (Morris Water Maze)."""

from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
import cv2
import os
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from .labyrinth import Labyrinth


class MorrisPool(Labyrinth):
	"""Experimento de Piscina de Morris (Morris Water Maze).

	La región de interés es un cuarto de círculo (cuadrante) que representa
	la zona donde se encuentra la plataforma sumergida. Las métricas incluyen
	trayectoria, distancia dentro/fuera de la región, tiempo en el cuadrante
	por evento individual y acumulado, y latencia al primer ingreso.
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
		"""Inicializa el experimento de Piscina de Morris.

		Args:
			video_path: Ruta al archivo de video o carpeta con videos.
			treatment: Nombre del tratamiento aplicado al sujeto.
			subject_id: Identificador del sujeto de experimentación.
			regions: ``RegionManager`` con exactamente una ``CircularFractionRegion``
				de 90° (cuarto de círculo).
			min_detection_area: Área mínima en píxeles para detectar al ratón.
			hitbox_size: Semilado de la hitbox cuadrada en píxeles.
			start_time: Tiempo en segundos desde el que se empieza a registrar.
			end_time: Tiempo en segundos en que termina el registro, o None.
			kernel_size: Tamaño del kernel morfológico del tracker.
			blur_size: Tamaño del kernel GaussianBlur. 0 desactiva el blur.

		Raises:
			ValueError: Si la región no es exactamente una ``CircularFractionRegion``
				de 90°.
		"""
		super().__init__(
			video_path, treatment, subject_id, "MorrisPool", regions,
			min_detection_area, hitbox_size, start_time, end_time,
			kernel_size, blur_size,
		)
		self.validate_morris_region()
		self.enter_frame = []
		self.exit_frame  = []
		self.event_list  = []

	# ------------------------------------------------------------------
	# Validaciones específicas
	# ------------------------------------------------------------------

	def validate_morris_region(self) -> None:
		"""Valida que la región sea compatible con Morris Pool.

		Raises:
			ValueError: Si no hay exactamente una región, si no es
				``CircularFractionRegion``, o si no cubre exactamente 90°.
		"""
		if len(self.regions.regions) != 1:
			raise ValueError(
				"Morris Pool requiere exactamente una region de interes "
				"(el cuadrante donde estuvo la plataforma)."
			)
		region = next(iter(self.regions.regions))

		if not isinstance(region, CircularFractionRegion):
			raise ValueError("La region de Morris Pool debe ser CircularFractionRegion.")

		angle_span = (region.angle_end - region.angle_start) % 360
		if not np.isclose(angle_span, 90.0, atol=1e-6):
			raise ValueError("La region de Morris Pool debe ser un cuarto de circulo (90 grados).")

	# ------------------------------------------------------------------
	# Visualización específica
	# ------------------------------------------------------------------

	def draw_arena_outline(self, canvas: np.ndarray) -> None:
		"""Dibuja el círculo completo de la piscina en gris oscuro sobre el canvas.

		Usa el centro y radio de la ``CircularFractionRegion`` definida como
		región de interés.

		Args:
			canvas: Imagen BGR sobre la cual dibujar.
		"""
		region = next(iter(self.regions.regions))
		center = tuple(map(int, region.center))
		radius = int(region.radius)
		cv2.circle(canvas, center, radius, (40, 40, 40), -1)
		cv2.circle(canvas, center, radius, (180, 180, 180), 2)

	def save_heatmap_image(
		self,
		all_trajectories: dict,
		first_frame: np.ndarray,
		output_dir: str,
	) -> None:
		"""Genera el heatmap específico para Morris Pool.

		El calor queda recortado dentro del círculo de la piscina. Incluye
		una barra de escala de temperatura (frío→caliente) en el margen derecho.
		El fondo es negro fuera del círculo.

		Args:
			all_trajectories: ``{nombre_video: (traj_x, traj_y)}``.
			first_frame: Primer frame de cualquier video para obtener dimensiones.
				Puede ser None.
			output_dir: Carpeta donde guardar el PNG.
		"""
		canvas, height, width = self.make_black_canvas(first_frame)

		region = next(iter(self.regions.regions))
		center = tuple(map(int, region.center))
		radius = int(region.radius)

		density_norm = self.compute_density(all_trajectories, height, width)
		if density_norm is None:
			return

		heatmap_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

		# Máscara circular — solo pintar dentro de la piscina
		circle_mask = np.zeros((height, width), dtype=np.uint8)
		cv2.circle(circle_mask, center, radius, 255, -1)

		# Fondo gris oscuro dentro del círculo, negro fuera
		cv2.circle(canvas, center, radius, (40, 40, 40), -1)

		# Aplicar heatmap solo donde hay densidad Y dentro del círculo
		heat_mask = (density_norm > 0) & (circle_mask == 255)
		canvas[heat_mask] = heatmap_color[heat_mask]

		# Contorno de la piscina y cuadrante en blanco
		cv2.circle(canvas, center, radius, (255, 255, 255), 2)
		for reg in self.regions.regions:
			reg.draw(canvas, (255, 255, 255), 2)

		# --- Barra de escala de temperatura ---
		bar_x      = width - 30
		bar_y_top  = 20
		bar_height = height - 40
		bar_width  = 18

		for i in range(bar_height):
			value     = int(255 * (1.0 - i / bar_height))  # arriba=caliente
			color_bar = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
			cv2.rectangle(
				canvas,
				(bar_x, bar_y_top + i),
				(bar_x + bar_width, bar_y_top + i + 1),
				color_bar.tolist(), -1,
			)

		cv2.rectangle(canvas, (bar_x, bar_y_top), (bar_x + bar_width, bar_y_top + bar_height), (255, 255, 255), 1)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(canvas, "Alto", (bar_x - 2, bar_y_top - 5),              font, 0.45, (255, 255, 255), 1)
		cv2.putText(canvas, "Bajo", (bar_x - 2, bar_y_top + bar_height + 14), font, 0.45, (255, 255, 255), 1)

		img_path = os.path.join(output_dir, f"heatmap_{self.mace_type}_{self.subject_id}_{self.treatment}.png")
		cv2.imwrite(img_path, canvas)
		print(f"Mapa de calor guardado en: {img_path}")

	# ------------------------------------------------------------------
	# Procesamiento
	# ------------------------------------------------------------------

	def process_frame(self, position: list, time: float) -> None:
		"""Almacena la posición del frame actual en la trayectoria.

		Args:
			position: Lista de coordenadas ``(x, y)`` detectadas en el frame.
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
			all_results: ``{nombre_video: events_on_each_region}``.
			all_trajectories: ``{nombre_video: (trajectory_x, trajectory_y)}``.
			all_video_paths: ``{nombre_video: ruta_absoluta}``.
			all_first_frames: ``{nombre_video: primer_frame}``.
			all_start_times: ``{nombre_video: start_time}``.
		"""
		self.write_results(all_results, all_trajectories, all_video_paths, all_first_frames, all_start_times)

	# ------------------------------------------------------------------
	# Métodos auxiliares (mantenidos por documentación histórica)
	# ------------------------------------------------------------------

	def get_time_index_in_out_of_region(self, events_on_each_region: dict) -> None:
		"""Obtiene los índices de entrada y salida de la región de interés.

		Nota: Método legado, mantenido por documentación histórica. La lógica
		equivalente está integrada en ``write_summary`` vía ``compute_region_metrics``.

		Args:
			events_on_each_region: ``{region_id: ZoneState}`` con los eventos
				de cada región.
		"""
		region_id        = next(iter(self.regions.regions)).region_id
		self.enter_frame = events_on_each_region[region_id].enter_frame
		self.exit_frame  = events_on_each_region[region_id].exit_frame
		self.check_enter_exit_frame_lists()

	def check_enter_exit_frame_lists(self) -> None:
		"""Verifica que las listas de entrada y salida tengan la misma longitud.

		Si hay una entrada sin salida correspondiente, agrega una salida artificial
		al final del video usando ``end_time``.

		Nota: Método legado, mantenido por documentación histórica.

		Raises:
			TypeError: Si ``self.end_time`` es None cuando se intenta calcular
				la salida artificial.
		"""
		if len(self.enter_frame) != len(self.exit_frame):
			print(f"Advertencia: entradas {len(self.enter_frame)} y salidas {len(self.exit_frame)} no coinciden.")
			if len(self.enter_frame) == len(self.exit_frame) + 1:
				print("Hay una entrada mas que salidas — se agrega salida al final del video.")
				self.exit_frame.append(int(self.fps * self.end_time))
			elif len(self.enter_frame) > len(self.exit_frame):
				print("Hay mas entradas que salidas. REVISAR LOGICA.")
			else:
				print("Hay mas salidas que entradas. REVISAR LOGICA.")

	def get_distance_and_time_inside_region(self) -> None:
		"""Calcula distancia y tiempo dentro de la región para cada evento.

		Nota: Método legado, mantenido por documentación histórica. La lógica
		equivalente está integrada en ``compute_region_metrics``.
		"""
		for i in range(len(self.enter_frame)):
			self.event_list.append([])
			self.event_list[i].append((self.exit_frame[i] - self.enter_frame[i]) / self.fps)
			self.event_list[i].append(self.get_total_distance(
				start_frame=self.enter_frame[i], end_frame=self.exit_frame[i]))

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
		"""Escribe el resumen de Morris Pool en la hoja de Excel.

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
		ws.append(["Nº de entradas a la región",    len(m["enter_frames"])])
		ws.append(["Tiempo total en región (s)",     round(m["total_time"], 3)])
		ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
		ws.append(["Latencia al primer ingreso (s)", round(m["latency"], 3) if m["latency"] is not None else "No entró"])
		ws.append(["% tiempo en región",             round(m["pct_time"], 2)])
		ws.append(["% distancia en región",          round(m["pct_distance"], 2)])
		ws.append([])
		self.write_event_table(ws, m)