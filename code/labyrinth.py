from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion
import numpy as np
from logic import EventLogic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

class Labyrinth:
	"""
	Clase abstracta para los diferentes laberintos
	"""
	def __init__(self, video_path: str, treatment: str, subject_id: str, mace_type: str, regions: list, min_detection_area: int, hitbox_size: int, start_time: float, end_time: float | None = None) :
		self.video_path = video_path
		self.treatment = treatment
		self.subject_id = subject_id
		self.mace_type = mace_type
		self.regions = regions
		self.start_time = start_time
		self.end_time = end_time
		self.min_detection_area = min_detection_area
		self.hitbox_size = hitbox_size
		self.fps = 0 # Este atributo se llenará al abrir el video, para que esté disponible en toda la clase y no haya que pasarlo como parámetro a cada función.

		self._validate_inputs()

		# Regiones y lógica de eventos
		#self.region_manager = regions
		#self.logic = EventLogic(self.region_manager)


		self.trajectory_x = []
		self.trajectory_y = []
		self.trajectory_time = []
		

		# Resultados
		self.results = {}
	
	#--------------------------------------
	#	Validaciones generales
	#--------------------------------------

	def _validate_inputs(self):
		if not isinstance(self.video_path, str):
			raise ValueError("La dirección del video (video_path) debe ser una cadena de texto.")
		if not isinstance(self.treatment, str):
			raise ValueError("El tratamiento (treatment) debe ser una cadena de texto.")
		if not isinstance(self.subject_id, str):
			raise ValueError("La identificación del sujeto de experimentación (subject_id) debe ser una cadena de texto.")
		if not isinstance(self.mace_type, str):
			raise ValueError("El tipo de laberinto (mace_type) debe ser una cadena de texto.")
		if not isinstance(self.regions, RegionManager):
			raise ValueError("Las regiones de interés (regions) debe ser un RegionManager.")
		if not isinstance(self.start_time, (int, float)):
			raise ValueError("start_time debe ser un número.")
		if self.end_time is None:
			print("Advertencia: end_time no especificado, se procesará todo el video desde start_time hasta el final.")
		else:
			if not isinstance(self.end_time, (int, float)):
				raise ValueError("end_time debe ser un número o None.")
			if self.start_time < 0 or self.end_time < 0:
				raise ValueError("start_time y end_time deben ser números no negativos.")
			if self.start_time >= self.end_time:
				raise ValueError("start_time debe ser menor que end_time.")
		if not isinstance(self.min_detection_area, int) or self.min_detection_area <= 0:
			raise ValueError("min_detection_area debe ser un entero positivo.")
		if not isinstance(self.hitbox_size, int) or self.hitbox_size <= 0:
			raise ValueError("hitbox_size debe ser un entero positivo.")
		if not isinstance(self.fps, int) or self.fps < 0:
			raise ValueError("fps debe ser un entero positivo.")
		

	def process_frame(self, position, time):
		raise NotImplementedError("La función process_video debe ser implementada por cada tipo específico de laberinto.")
	
	def process_video(self):
		raise NotImplementedError("La función process_video debe ser implementada por cada tipo específico de laberinto.")

	def write_results(self):
		raise NotImplementedError("La función write_results debe ser implementada por cada tipo específico de laberinto.")

	def get_position(self, position, time):
		"""
		Recoje los datos de posición y tiempo para obtenidos en cada frame, y los almacena en listas para su posterior análisis.

		Parámetros
		----------
		position : tuple
			Coordenadas (x, y) del ratón detectado en el frame actual.
		time : float
			Tiempo actual en segundos.
		
		Retorna
		-------
		None
		"""
		if len(position) == 0:
			return # No se detectó posición en este frame, no se agrega nada a la trayectoria	
		x,y = position[-1]
		t = time
		self.trajectory_x.append(x)
		self.trajectory_y.append(y)
		self.trajectory_time.append(t)



	def get_total_distance(self, start_frame=0, end_frame=None):
		"""
		Calcula la distancia total recorrida en píxeles dentro de un rango de frames.

		Los frames de entrada son absolutos del video. Internamente se convierten
		a índices relativos de la trayectoria, que empieza en start_time * fps.

		Parámetros
		----------
		start_frame : int
			Frame absoluto del video desde el cual empezar (default: 0).
		end_frame : int or None
			Frame absoluto del video hasta el cual contar (default: None = hasta el final).
		
		Retorna
		-------
		float
			Distancia total en píxeles recorrida entre start_frame y end_frame.
			Retorna 0.0 si no hay suficientes puntos.
		"""
		trajectory = list(zip(self.trajectory_x, self.trajectory_y))

		if len(trajectory) <= 1:
			return 0.0

		# Convertir frames absolutos a índices relativos de la trayectoria
		recording_start = int(self.start_time * self.fps)

		if end_frame is None:
			end_frame = len(trajectory)
		else:
			end_frame = max(0, end_frame - recording_start)

		start_frame = max(0, start_frame - recording_start)

		# Limitar a rango válido
		end_frame = min(len(trajectory), end_frame)

		trajectory_filtered = trajectory[start_frame:end_frame]

		total_distance = 0.0
		for i in range(1, len(trajectory_filtered)):
			pt1 = trajectory_filtered[i - 1]
			pt2 = trajectory_filtered[i]
			dx = pt2[0] - pt1[0]
			dy = pt2[1] - pt1[1]
			total_distance += np.sqrt(dx * dx + dy * dy)

		return total_distance


class MorrisPool(Labyrinth):
	"""
	Clase para los laberientos de Piscina de Morris
	- Se define una región de un cuarto de circulo (cuadrante) como región de interés, el cuadrante dónde se encuentra la plataforma.
	- Se requiere obtener la trayectoria del sujeto de experimentación
	- Se requiere la distancia recorrida dentro y fuera de la región de interés
	- Se requiere el tiempo dentro del cuadrante de interés (por evento individual ---> entrada y salida, y acumulado total)
	"""

	def __init__(self, video_path: str, treatment: str, subject_id: str, regions: list, min_detection_area: int, hitbox_size: int, start_time: float, end_time: float | None = None) :
		
		super().__init__(video_path, treatment, subject_id,"MorrisPool", regions, min_detection_area, hitbox_size, start_time , end_time)


		self._validate_morris_region()
		self.enter_frame = []
		self.exit_frame = []
		self.event_list = []

	#-------------------------------------------
	# Validaciones específicas para Morris Pool
	#-------------------------------------------
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
			raise ValueError(
				"La región de Morris Pool debe ser CircularFractionRegion."
			)

		# 3) Verificar que sea 1/4 de círculo
		# Usamos diferencia angular en lugar de confiar en 'fraction'
		angle_span = (region.angle_end - region.angle_start) % 360

		if not np.isclose(angle_span, 90.0, atol=1e-6):
			raise ValueError(
				"La región de Morris Pool debe ser un cuarto de círculo (90°)."
			)

	#--------------------------------------
	# Resultados específicos para Morris Pool
	#--------------------------------------

	def process_frame(self, position, time):
		"""
		Procesaun frame a la vez para extraer la trayectoria del sujeto, los eventos de entrada y salida de la región de interés, y calcula la distancia recorrida dentro de la región.
		
		Parámetros
		----------
		position : tuple
			Coordenadas (x, y) del ratón detectado en el frame actual.
		time : float
			Tiempo actual en segundos.

		Retorna
		-------
		None, pero actualiza los atributos de la clase relacionados con la trayectoria.
		"""
		self.get_position(position, time)
		


	def process_video(self, events_on_each_region):
		"""
		Procesa el video completo para extraer la trayectoria del sujeto, los eventos de entrada y salida de la región de interés, y calcula la distancia recorrida dentro de la región.
		"""
		self.get_time_index_in_out_of_region(events_on_each_region)
		self.get_distance_and_time_inside_region()
		self.write_results()

	
	def get_time_index_in_out_of_region(self, events_on_each_region):
		"""
		Funcion que obtiene los indices de entrada y salida de la región de interés, a partir de la lista de eventos de cada región.


		Parámetros
		----------
		time : float
			Tiempo actual en segundos, este viene del loop principal del experimento, y se usa para verificar si ya se han registrado todos los eventos de entrada y salida antes de calcular la distancia recorrida dentro y fuera de la región.
		events_on_each_region : dict
			Es el atributo 'states' de la clase EventLogic, que es un diccionario que mapea region_id -> ZoneState, y cada ZoneState tiene una lista de frames de entrada y salida (enter_frame y exit_frame) que se van llenando a medida que se actualiza la lógica de eventos en cada frame.
		
		Retorna
		-------
		None, pero actualiza los atributos self.entries_idx y self.exits_idx con las listas de frames de entrada y salida respectivamente, para la región de interés (en este caso, la única región del Morris Pool).
		"""

		region_id = next(iter(self.regions.regions)).region_id # Solo hay una región en Morris Pool
		self.enter_frame = events_on_each_region[region_id].enter_frame
		self.exit_frame = events_on_each_region[region_id].exit_frame

		self.check_enter_exit_frame_lists()
		

	def check_enter_exit_frame_lists(self):
		"""
		Esta función verifica que las listas de frames de entrada y salida tengan la misma longitud, lo cual es necesario para calcular correctamente la distancia recorrida dentro y fuera de la región. Si hay una discrepancia, se emite una advertencia y se intenta corregir agregando una salida al final del video si hay una entrada sin salida correspondiente. Esta función se llama al final del video para asegurarse de que se hayan registrado todos los eventos antes de realizar los cálculos finales.

		Parâmetros
		----------
		No tiene parámetros, ya que accede directamente a los atributos de la clase relacionados con los eventos de entrada y salida.
			- self.entries_idx: Lista de frames de entrada a la región de interés.
			- self.exits_idx: Lista de frames de salida de la región de interés.
		Retorna
		-------
		No retorna ningún valor, pero modifica las listas de frames de entrada y salida si es necesario para asegurar que tengan la misma longitud, lo cual es crucial para los cálculos posteriores de distancia recorrida dentro y fuera de la región.
			- Si hay una entrada sin salida correspondiente, se agrega una salida al final del video para corregir la discrepancia.
			- Si hay más entradas que salidas o viceversa, se emite una advertencia indicando un posible error en la detección o en la lógica de eventos, lo cual debería ser revisado.
			- Si las listas ya tienen la misma longitud, no se realiza ninguna acción adicional.
		"""
		if len(self.enter_frame) != len(self.exit_frame):
			print(f"Advertencia: El número de entradas {len(self.enter_frame)} y salidas {len(self.exit_frame)} no coincide.")
			if len(self.enter_frame) == len(self.exit_frame) + 1:
				print("Hay una entrada más que salidas. Esto podría indicar que el sujeto entró a la región pero no salió antes de que terminara el video.")
				self.exit_frame.append(int(self.fps*self.end_time)) # Agregar una salida al final del video
			elif len(self.enter_frame) > len(self.exit_frame):
				print("Hay más entradas que salidas. Esto podría indicar un error en la detección o que el sujeto entró a la región sin haber salido correctamente. REVISAR LOGICAS")
			else:
				print("Hay más salidas que entradas. Esto podría indicar un error en la detección o que el sujeto salió de la región sin haber entrado correctamente. REVISAR LOGICAS")

	def get_distance_and_time_inside_region(self):
		"""
		Obtiene una lista de todos los eventos individuales de entrada y salida de la región de interés, con su respectiva distancia recorrida dentro de la región y el tiempo dentro de la región para cada evento.

		Parámetros
		----------
		No tiene parámetros, ya que accede directamente a los atributos de la clase relacionados con los eventos de entrada y salida, así como a la trayectoria registrada durante el experimento.
			- self.enter_frame: Lista de frames de entrada a la región de interés.
			- self.exit_frame: Lista de frames de salida de la región de interés.
			- self.trajectory_x y self.trajectory_y: Listas de coordenadas x e y de la trayectoria del sujeto a lo largo del tiempo, registradas durante el experimento.

		Retorna
		-------
		event_list : list of lists
			Una lista donde cada elemento es una sublista que contiene:
				- Tiempo dentro de la región para ese evento (en segundos).
				- Distancia recorrida dentro de la región para ese evento (en píxeles).
		"""

		
		for i in range(len(self.enter_frame)):
			self.event_list.append([])
			self.event_list[i].append((self.exit_frame[i] - self.enter_frame[i])/self.fps) # agrego el tiempo dentro de la region	
			self.event_list[i].append( self.get_total_distance(start_frame=self.enter_frame[i], end_frame=self.exit_frame[i])) # agrego la distancia recorrida dentro de la region
	

	def write_results(self):
		"""
		Genera un archivo Excel con los resultados del experimento de Morris Pool.

		El archivo contiene tres secciones:
		- Metadatos: información del sujeto, tratamiento, laberinto y parámetros de grabación.
		- Resumen: métricas globales del experimento (entradas, tiempos, distancias, latencia, porcentajes).
		- Detalle por evento: tabla con cada entrada/salida individual a la región de interés,
		incluyendo frames, tiempos, duración y distancia recorrida dentro de la región.

		El archivo se guarda en el directorio de trabajo con el nombre:
			results_{mace_type}_{subject_id}_{treatment}.xlsx

		Retorna
		-------
		None
		"""
		
		# ------------------------------------------------------------------
		# Cálculos previos a escribir resultados
		# ------------------------------------------------------------------

		# Tiempo total que el sujeto pasó dentro de la región (suma de todos los eventos)
		total_time_in_region = sum(e[0] for e in self.event_list)

		# Distancia total recorrida durante toda la grabación
		total_distance = self.get_total_distance()

		# Duración total de la grabación en segundos
		total_recording_time = len(self.trajectory_x) / self.fps

		# Latencia: segundos desde start_recording hasta el primer ingreso a la región
		latency = (self.enter_frame[0] / self.fps) - self.start_time if self.enter_frame else None

		# Porcentaje del tiempo de grabación que el sujeto pasó en la región
		pct_time = (total_time_in_region / total_recording_time * 100) if total_recording_time > 0 else 0

		# Porcentaje de la distancia total recorrida que ocurrió dentro de la región
		total_distance_in_region = sum(e[1] for e in self.event_list)
		pct_distance = (total_distance_in_region / total_distance * 100) if total_distance > 0 else 0

		# ------------------------------------------------------------------
		# Crear workbook
		# ------------------------------------------------------------------
		wb = Workbook()
		ws = wb.active
		ws.title = "Resultados"

		# ------------------------------------------------------------------
		# Sección 1: Metadatos del experimento
		# ------------------------------------------------------------------
		meta = [
			("Sujeto",         self.subject_id),
			("Tratamiento",    self.treatment),
			("Laberinto",      self.mace_type),
			("Start time (s)", self.start_time),
			("End time (s)",   self.end_time if self.end_time else "Hasta el final"),
			("FPS",            self.fps),
		]
		for row in meta:
			ws.append(row)

		ws.append([])  # separador

		# ------------------------------------------------------------------
		# Sección 2: Resumen global
		# ------------------------------------------------------------------
		ws.append(["RESUMEN", ""])
		ws.append(["Nº de entradas a la región",    len(self.enter_frame)])
		ws.append(["Tiempo total en región (s)",     round(total_time_in_region, 3)])
		ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
		ws.append(["Latencia al primer ingreso (s)", round(latency, 3) if latency is not None else "No entró"])
		ws.append(["% tiempo en región",             round(pct_time, 2)])
		ws.append(["% distancia en región",          round(pct_distance, 2)])

		ws.append([])  # separador

		# ------------------------------------------------------------------
		# Sección 3: Detalle por evento (una fila por entrada/salida)
		# ------------------------------------------------------------------
		headers = [
			"Evento #",
			"Frame entrada",
			"Tiempo entrada (s)",
			"Frame salida",
			"Tiempo salida (s)",
			"Duración en región (s)",
			"Distancia en región (px)",
		]
		ws.append(headers)

		# Estilo de encabezados: fondo azul, texto blanco, centrado
		header_row = ws.max_row
		for col in range(1, len(headers) + 1):
			cell = ws.cell(row=header_row, column=col)
			cell.font = Font(bold=True, color="FFFFFF")
			cell.fill = PatternFill("solid", start_color="2F5496")
			cell.alignment = Alignment(horizontal="center")

		# Una fila por cada evento de entrada/salida
		for i, event in enumerate(self.event_list):
			ws.append([
				i + 1,
				self.enter_frame[i],
				round(self.enter_frame[i] / self.fps, 3),
				self.exit_frame[i],
				round(self.exit_frame[i] / self.fps, 3),
				round(event[0], 3),   # duración dentro de la región
				round(event[1], 2),   # distancia dentro de la región
			])

		# Fila de totales con fórmulas Excel (se recalculan al abrir el archivo)
		first_data_row = header_row + 1
		last_data_row = ws.max_row
		ws.append([
			"TOTAL", "", "", "", "",
			f"=SUM(F{first_data_row}:F{last_data_row})",
			f"=SUM(G{first_data_row}:G{last_data_row})",
		])
		total_row = ws.max_row
		for col in range(1, len(headers) + 1):
			cell = ws.cell(row=total_row, column=col)
			cell.font = Font(bold=True)
			cell.fill = PatternFill("solid", start_color="D9E1F2")

		# --- Centrar todo el contenido ---
		for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
			for cell in row:
				cell.alignment = Alignment(horizontal="center")
        
		# ------------------------------------------------------------------
		# Formato: ancho de columnas
		# ------------------------------------------------------------------
		col_widths = [10, 16, 20, 14, 18, 24, 26]
		for i, width in enumerate(col_widths, 1):
			ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

		# ------------------------------------------------------------------
		# Guardar archivo
		# ------------------------------------------------------------------
		filename = f"results_{self.mace_type}_{self.subject_id}_{self.treatment}.xlsx"
		wb.save(filename)
		print(f"Resultados guardados en: {filename}")
  
class CrossMaze(Labyrinth):
    """
    Clase para laberintos en cruz.
    - Soporta cualquier número de regiones poligonales (brazos).
    - Métricas por región: latencia, entradas, tiempo, distancia, porcentajes.
    """

    def __init__(self, video_path, treatment, subject_id, regions,
                 min_detection_area, hitbox_size, start_time, end_time=None):

        super().__init__(video_path, treatment, subject_id, "CrossMaze",
                         regions, min_detection_area, hitbox_size, start_time, end_time)

        self._validate_cross_maze_regions()

    def _validate_cross_maze_regions(self):
        if len(self.regions.regions) < 2:
            raise ValueError("CrossMaze requiere al menos 2 regiones de interés.") # De momento, luego preguntar si se quiere exigir exactamente otra cantidad de regiones como minimo.
        for region in self.regions.regions:
            if not isinstance(region, PolygonRegion):
                raise ValueError("Todas las regiones de CrossMaze deben ser PolygonRegion.")

    def process_frame(self, position, time):
        self.get_position(position, time)

    def process_video(self, events_on_each_region):
        self.write_results(events_on_each_region)

    def write_results(self, events_on_each_region):
        """
        Genera un archivo Excel con los resultados del experimento CrossMaze.

        Contiene:
        - Metadatos del experimento.
        - Resumen global: distancia total y duración total de grabación.
        - Por cada región: resumen (latencia, entradas, tiempo, distancia, porcentajes)
          y tabla de detalle por evento.

        Parámetros
        ----------
        events_on_each_region : dict
            Atributo 'states' de EventLogic — mapea region_id -> ZoneState.
        """
        # ------------------------------------------------------------------
        # Cálculos globales
        # ------------------------------------------------------------------
        total_distance = self.get_total_distance()
        total_recording_time = len(self.trajectory_x) / self.fps

        # ------------------------------------------------------------------
        # Crear workbook
        # ------------------------------------------------------------------
        wb = Workbook()
        ws = wb.active
        ws.title = "Resultados"

        # ------------------------------------------------------------------
        # Sección 1: Metadatos
        # ------------------------------------------------------------------
        meta = [
            ("Sujeto",         self.subject_id),
            ("Tratamiento",    self.treatment),
            ("Laberinto",      self.mace_type),
            ("Start time (s)", self.start_time),
            ("End time (s)",   self.end_time if self.end_time else "Hasta el final"),
            ("FPS",            self.fps),
        ]
        for row in meta:
            ws.append(row)
        ws.append([])

        # ------------------------------------------------------------------
        # Sección 2: Resumen global
        # ------------------------------------------------------------------
        ws.append(["RESUMEN GLOBAL", ""])
        ws.append(["Distancia total recorrida (px)", round(total_distance, 2)])
        ws.append(["Duración total grabación (s)",   round(total_recording_time, 3)])
        ws.append([])

        # ------------------------------------------------------------------
        # Sección 3: Por cada región
        # ------------------------------------------------------------------
        for region in self.regions.regions:
            region_id = region.region_id
            state = events_on_each_region[region_id]

            enter_frames = state.enter_frame
            exit_frames  = state.exit_frame

            # Verificar listas de entrada/salida
            if len(enter_frames) == len(exit_frames) + 1:
                exit_frames.append(int(self.fps * (self.end_time or total_recording_time + self.start_time)))

            # Calcular métricas de esta región
            event_list = []
            for i in range(len(enter_frames)):
                duration = (exit_frames[i] - enter_frames[i]) / self.fps
                distance = self.get_total_distance(start_frame=enter_frames[i], end_frame=exit_frames[i])
                event_list.append((duration, distance))

            total_time_in_region     = sum(e[0] for e in event_list)
            total_distance_in_region = sum(e[1] for e in event_list)
            latency = (enter_frames[0] / self.fps) - self.start_time if enter_frames else None
            pct_time     = (total_time_in_region / total_recording_time * 100)     if total_recording_time > 0 else 0
            pct_distance = (total_distance_in_region / total_distance * 100) if total_distance > 0 else 0

            # --- Resumen de esta región ---
            ws.append([f"REGIÓN: {region_id}", ""])
            ws.append(["Nº de entradas",                len(enter_frames)])
            ws.append(["Tiempo total en región (s)",     round(total_time_in_region, 3)])
            ws.append(["Distancia en región (px)",       round(total_distance_in_region, 2)])
            ws.append(["Latencia al primer ingreso (s)", round(latency, 3) if latency is not None else "No entró"])
            ws.append(["% tiempo en región",             round(pct_time, 2)])
            ws.append(["% distancia en región",          round(pct_distance, 2)])
            ws.append([])

            # --- Tabla de detalle por evento ---
            headers = [
                "Evento #",
                "Frame entrada",
                "Tiempo entrada (s)",
                "Frame salida",
                "Tiempo salida (s)",
                "Duración en región (s)",
                "Distancia en región (px)",
            ]
            ws.append(headers)

            header_row = ws.max_row
            for col in range(1, len(headers) + 1):
                cell = ws.cell(row=header_row, column=col)
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.fill      = PatternFill("solid", start_color="2F5496")
                cell.alignment = Alignment(horizontal="center")

            for i, event in enumerate(event_list):
                ws.append([
                    i + 1,
                    enter_frames[i],
                    round(enter_frames[i] / self.fps, 3),
                    exit_frames[i],
                    round(exit_frames[i] / self.fps, 3),
                    round(event[0], 3),
                    round(event[1], 2),
                ])

            # Fila de totales
            first_data_row = header_row + 1
            last_data_row  = ws.max_row
            ws.append([
                "TOTAL", "", "", "", "",
                f"=SUM(F{first_data_row}:F{last_data_row})",
                f"=SUM(G{first_data_row}:G{last_data_row})",
            ])
            total_row = ws.max_row
            for col in range(1, len(headers) + 1):
                cell = ws.cell(row=total_row, column=col)
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", start_color="D9E1F2")

            ws.append([])  # separador entre regiones

        # ------------------------------------------------------------------
        # Centrar todo
        # ------------------------------------------------------------------
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.alignment = Alignment(horizontal="center")

        # ------------------------------------------------------------------
        # Ancho de columnas
        # ------------------------------------------------------------------
        col_widths = [10, 16, 20, 14, 18, 24, 26]
        for i, width in enumerate(col_widths, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

        # ------------------------------------------------------------------
        # Guardar
        # ------------------------------------------------------------------
        filename = f"results_{self.mace_type}_{self.subject_id}_{self.treatment}.xlsx"
        wb.save(filename)
        print(f"Resultados guardados en: {filename}")