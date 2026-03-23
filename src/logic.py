"""Lógica de eventos de entrada y salida en regiones de interés.

Este módulo es independiente de OpenCV y de cualquier interfaz gráfica.
Recibe posiciones y tiempos, y mantiene el historial de cada región.
"""

import math


class ZoneState:
	"""Estado temporal asociado a una región de interés.

	Attributes:
		inside: True si el objeto está actualmente dentro de la región.
		enter_time: Timestamp en segundos del último ingreso, o None si
			el objeto no está dentro.
		enter_frame: Lista de frames absolutos en los que ocurrió una entrada.
		exit_frame: Lista de frames absolutos en los que ocurrió una salida.
		total_time: Tiempo total acumulado en segundos dentro de la región.
		entries: Número total de entradas a la región.
		events: Lista de tuplas (enter_time, exit_time) en segundos.
		_last_position: Última posición registrada dentro de la región.
		_current_distance: Distancia acumulada en píxeles durante el evento actual.
	"""

	def __init__(self):
		"""Inicializa el estado de una región con todos los contadores en cero."""
		self.inside: bool = False
		self.enter_time: float | None = None
		self.enter_frame: list[int] = []
		self.exit_frame: list[int] = []
		self.total_time: float = 0.0
		self.entries: int = 0
		self.events: list[tuple[float, float]] = []
		self._last_position: tuple | None = None
		self._current_distance: float = 0.0


class EventLogic:
	"""Lógica de eventos de entrada y salida para múltiples regiones.

	Recibe posiciones del objeto a lo largo del tiempo y determina
	eventos de entrada y salida en cada región de interés.

	Un evento se descarta al salir si la distancia acumulada dentro de la
	región es menor a 1 píxel Y la duración es menor a min_entry_time.
	Si la duración supera min_entry_time el evento se conserva aunque la
	distancia sea insignificante (caso de tracker perdido dentro de la región).

	Attributes:
		regions: Lista de regiones de interés.
		hitbox_w: Semiancho de la hitbox cuadrada en píxeles.
		hitbox_h: Semialto de la hitbox cuadrada en píxeles.
		min_entry_time: Tiempo mínimo en segundos para conservar un evento
			con distancia insignificante.
		states: Diccionario {region_id: ZoneState} con el estado de cada región.
	"""

	def __init__(self, region_manager, hitbox_size: int = 0, min_entry_time: float = 1.0):
		"""Inicializa la lógica de eventos.

		Args:
			region_manager: Objeto RegionManager con la lista de regiones.
			hitbox_size: Semilado de la hitbox cuadrada en píxeles.
			min_entry_time: Tiempo mínimo en segundos para conservar un evento
				con distancia menor a 1 píxel.
		"""
		self.regions = region_manager.regions
		self.hitbox_w = hitbox_size
		self.hitbox_h = hitbox_size
		self.min_entry_time = min_entry_time
		self.states = {r.region_id: ZoneState() for r in self.regions}

	def update(self, position, t: float, frame_idx: int) -> None:
		"""Actualiza el estado de todas las regiones dado un nuevo frame.

		Args:
			position: Posición (x, y) del objeto detectado. Si es None,
				no se actualiza ningún estado.
			t: Timestamp actual en segundos (frame_idx / fps).
			frame_idx: Índice del frame actual, para registro de eventos.
		"""
		if position is None:
			return

		for region in self.regions:
			state = self.states[region.region_id]
			inside_now = region.contains_hitbox(position, self.hitbox_w, self.hitbox_h)

			# Evento de entrada
			if inside_now and not state.inside:
				state.inside = True
				state.enter_time = t
				state.enter_frame.append(frame_idx)
				state.entries += 1
				state._last_position = position
				state._current_distance = 0.0

			# Acumular distancia mientras el ratón está dentro
			elif inside_now and state.inside:
				if state._last_position is not None:
					dx = position[0] - state._last_position[0]
					dy = position[1] - state._last_position[1]
					state._current_distance += math.sqrt(dx * dx + dy * dy)
				state._last_position = position

			# Evento de salida
			elif not inside_now and state.inside:
				state.inside = False
				state._last_position = None

				duration = t - state.enter_time  # type: ignore

				# Descartar si distancia insignificante Y duración menor al mínimo
				if state._current_distance < 10.0 and duration < self.min_entry_time:
					state.enter_frame.pop()
					state.entries -= 1
					state.enter_time = None
					state._current_distance = 0.0
					continue

				state.total_time += t - state.enter_time  # type: ignore
				state.exit_frame.append(frame_idx)
				state.events.append((state.enter_time, t))  # type: ignore
				state.enter_time = None
				state._current_distance = 0.0