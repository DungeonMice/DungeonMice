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

		self.validate_morris_region()
		self.enter_frame = []
		self.exit_frame  = []
		self.event_list  = []

	# ----------------------------------------------------------------------
	# Validaciones específicas para Morris Pool
	# ----------------------------------------------------------------------

	def validate_morris_region(self):
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
	# Visualización específica: círculo de la piscina como fondo
	# ----------------------------------------------------------------------

	def draw_arena_outline(self, canvas):
		"""
		Dibuja el círculo completo de la piscina en gris oscuro sobre el canvas.
		Usa el centro y radio de la única región CircularFractionRegion definida.
		"""
		region = next(iter(self.regions.regions))
		center = tuple(map(int, region.center))
		radius = int(region.radius)
		cv2.circle(canvas, center, radius, (40, 40, 40), -1)
		cv2.circle(canvas, center, radius, (180, 180, 180), 2)

	def save_heatmap_image(self, all_trajectories, first_frame, output_dir):
		"""
		Heatmap específico para Morris Pool:
		- El calor queda recortado dentro del círculo de la piscina.
		- Se dibuja una barra de escala de temperatura (frío→caliente).
		- Fondo negro fuera del círculo.
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
		bar_x      = width - 30    # posición horizontal de la barra
		bar_y_top  = 20            # margen superior
		bar_height = height - 40   # altura de la barra
		bar_width  = 18

		# Gradiente vertical de 0 (frío, abajo) a 255 (caliente, arriba)
		for i in range(bar_height):
			value     = int(255 * (1.0 - i / bar_height))  # arriba=caliente
			color_bar = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
			cv2.rectangle(
				canvas,
				(bar_x, bar_y_top + i),
				(bar_x + bar_width, bar_y_top + i + 1),
				color_bar.tolist(), -1
			)

		# Borde de la barra
		cv2.rectangle(canvas, (bar_x, bar_y_top), (bar_x + bar_width, bar_y_top + bar_height), (255, 255, 255), 1)

		# Etiquetas "Alto" y "Bajo"
		font       = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.45
		thickness  = 1
		cv2.putText(canvas, "Alto", (bar_x - 2, bar_y_top - 5),       font, font_scale, (255, 255, 255), thickness)
		cv2.putText(canvas, "Bajo", (bar_x - 2, bar_y_top + bar_height + 14), font, font_scale, (255, 255, 255), thickness)

		img_path = os.path.join(output_dir, f"heatmap_{self.mace_type}_{self.subject_id}_{self.treatment}.png")
		cv2.imwrite(img_path, canvas)
		print(f"Mapa de calor guardado en: {img_path}")

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

	def write_summary(self, ws, events_on_each_region, total_distance, total_recording_time):
		"""Resumen de Morris Pool: una sola región."""
		region_id    = next(iter(self.regions.regions)).region_id
		state        = events_on_each_region[region_id]
		enter_frames = list(state.enter_frame)
		exit_frames  = list(state.exit_frame)
		enter_times  = list(state.events)  # lista de (enter_time, exit_time
		m = self.compute_region_metrics(
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
		self.write_event_table(ws, m)