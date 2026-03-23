"""Definición de regiones de interés (ROI) geométricas.

Este módulo define una interfaz común para regiones y varias implementaciones:

- ``PolygonRegion``: regiones definidas por polígonos (incluye rectángulos).
- ``CircleRegion``: regiones definidas por círculos.
- ``CircularFractionRegion``: sectores de círculo.

Cada región implementa:

- ``contains(point)``: prueba de pertenencia por punto central.
- ``overlap_fraction(point, hitbox_size)``: fracción del área de la hitbox dentro de la región.
- ``contains_hitbox(point, hitbox_size)``: decisión final según overlap_threshold.
- ``mask(shape)``: generación de máscara binaria.
- ``draw(frame)``: dibujo sobre un frame.
"""

import numpy as np
import cv2
import math

_OVERLAP_GRID_STEPS = 5  # Resolución de la grilla de muestreo (5×5 = 25 puntos)


class Region:
	"""Clase base abstracta para una región de interés (ROI).

	Todas las regiones geométricas deben implementar esta interfaz.
	No maneja tiempo, eventos ni lógica externa.

	Attributes:
		overlap_threshold: Fracción mínima del área de la hitbox que debe
			estar dentro de la región para considerarla como "dentro".

			- ``0.0`` (default): modo punto — se usa solo el centro de la hitbox.
			- ``0.0 < umbral <= 1.0``: modo área — se evalúa la fracción del área
			  de la hitbox que interseca con la región.
	"""

	def __init__(self, overlap_threshold: float = 0.0):
		"""Inicializa la región con su umbral de solapamiento.

		Args:
			overlap_threshold: Fracción mínima de la hitbox dentro de la región
				para activar la detección. 0.0 usa el modo punto original.

		Raises:
			ValueError: Si overlap_threshold no está en [0.0, 1.0].
		"""
		if not (0.0 <= overlap_threshold <= 1.0):
			raise ValueError("overlap_threshold debe estar en [0.0, 1.0]")
		self.overlap_threshold = overlap_threshold

	def contains(self, point: tuple[float, float]) -> bool:
		"""Determina si un punto pertenece a la región.

		Args:
			point: Punto en coordenadas de imagen ``(x, y)``.

		Returns:
			True si el punto está dentro de la región.
		"""
		raise NotImplementedError

	def overlap_fraction(self, point: tuple[float, float], hitbox_size: int) -> float:
		"""Calcula la fracción del área de la hitbox que interseca con la región.

		Usa una grilla de puntos uniformemente distribuidos dentro de la hitbox
		rectangular centrada en ``point``. La fracción es el cociente entre los
		puntos dentro de la región y el total de puntos de la grilla.

		Args:
			point: Centro de la hitbox en coordenadas de imagen ``(x, y)``.
			hitbox_size: Semilado de la hitbox cuadrada en píxeles.

		Returns:
			Fracción en ``[0.0, 1.0]``. 0.0 si no hay solapamiento, 1.0 si la
			hitbox está completamente dentro de la región.
		"""
		cx, cy = point
		steps = _OVERLAP_GRID_STEPS
		inside = 0
		total = steps * steps

		for i in range(steps):
			for j in range(steps):
				# Distribuir puntos de forma uniforme dentro de la hitbox,
				# con un margen de medio paso para evitar evaluar justo el borde.
				t_x = (i + 0.5) / steps  # [0, 1)
				t_y = (j + 0.5) / steps
				px = cx - hitbox_size + t_x * 2 * hitbox_size
				py = cy - hitbox_size + t_y * 2 * hitbox_size
				if self.contains((px, py)):
					inside += 1

		return inside / total

	def contains_hitbox(self, point: tuple[float, float], hitbox_size: int) -> bool:
		"""Determina si la hitbox está suficientemente dentro de la región.

		Si ``overlap_threshold`` es 0.0, delega en ``contains(point)``
		preservando el comportamiento original basado en el punto central.
		En caso contrario, calcula ``overlap_fraction`` y lo compara con
		el umbral configurado.

		Args:
			point: Centro de la hitbox en coordenadas de imagen ``(x, y)``.
			hitbox_size: Semilado de la hitbox cuadrada en píxeles.

		Returns:
			True si la condición de solapamiento se cumple.
		"""
		if self.overlap_threshold == 0.0:
			return self.contains(point)
		return self.overlap_fraction(point, hitbox_size) >= self.overlap_threshold

	def mask(self, shape: tuple[int, int]) -> np.ndarray:
		"""Genera una máscara binaria de la región.

		Args:
			shape: Dimensiones ``(alto, ancho)`` de la imagen.

		Returns:
			Imagen uint8 con 255 dentro de la región y 0 fuera.
		"""
		raise NotImplementedError

	def draw(
		self,
		frame: np.ndarray,
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
	) -> None:
		"""Dibuja la región sobre un frame.

		Args:
			frame: Imagen sobre la cual dibujar.
			color: Color del contorno en formato ``(B, G, R)``.
			thickness: Grosor del contorno en píxeles.
		"""
		raise NotImplementedError


class PolygonRegion(Region):
	"""Región de interés (ROI) definida por un polígono.

	Representa cualquier región poligonal: rectángulos, polígonos convexos
	y polígonos cóncavos. Los puntos se almacenan en formato compatible con
	OpenCV: ``(N, 1, 2)``.

	Attributes:
		region_id: Identificador único de la región.
		points: Vértices del polígono en formato OpenCV ``(N, 1, 2)``.
		overlap_threshold: Heredado de ``Region``.
	"""

	def __init__(self, region_id, points, overlap_threshold: float = 0.0):
		"""Inicializa una región poligonal.

		Args:
			region_id: Identificador único de la región.
			points: Vértices del polígono en orden como iterable de ``(x, y)``.
			overlap_threshold: Fracción mínima de solapamiento. Ver ``Region``.

		Raises:
			ValueError: Si hay menos de 3 puntos o overlap_threshold inválido.
		"""
		super().__init__(overlap_threshold)
		self.region_id = region_id
		self.points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
		if self.points.shape[0] < 3:
			raise ValueError("Un poligono debe tener al menos 3 puntos")

	def contains(self, point: tuple[float, float]) -> bool:
		"""Determina si un punto está dentro o sobre el borde del polígono.

		Args:
			point: Punto en coordenadas de imagen ``(x, y)``.

		Returns:
			True si está dentro o en el borde.
		"""
		pt = (float(point[0]), float(point[1]))
		return cv2.pointPolygonTest(self.points, pt, False) >= 0

	def mask(self, shape: tuple[int, int]) -> np.ndarray:
		"""Genera una máscara binaria del polígono.

		Args:
			shape: Dimensiones de la imagen ``(alto, ancho)``.

		Returns:
			Máscara binaria uint8.
		"""
		mask = np.zeros(shape, dtype=np.uint8)
		cv2.drawContours(mask, [self.points], -1, 255, -1)
		return mask

	def draw(
		self,
		frame: np.ndarray,
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
	) -> None:
		"""Dibuja el contorno del polígono sobre un frame.

		Args:
			frame: Imagen destino.
			color: Color del contorno en formato ``(B, G, R)``.
			thickness: Grosor de línea en píxeles.
		"""
		cv2.polylines(frame, [self.points], True, color, thickness)


class CircleRegion(Region):
	"""Región de interés (ROI) definida por un círculo.

	Attributes:
		region_id: Identificador único de la región.
		center: Centro del círculo como ``(x, y)``.
		radius: Radio en píxeles.
		overlap_threshold: Heredado de ``Region``.
	"""

	def __init__(self, region_id, center, radius, overlap_threshold: float = 0.0):
		"""Inicializa una región circular.

		Args:
			region_id: Identificador único.
			center: Centro del círculo como ``(x, y)``.
			radius: Radio en píxeles.
			overlap_threshold: Fracción mínima de solapamiento. Ver ``Region``.

		Raises:
			ValueError: Si el radio no es positivo o overlap_threshold inválido.
		"""
		super().__init__(overlap_threshold)
		self.region_id = region_id
		self.center = (float(center[0]), float(center[1]))
		self.radius = float(radius)
		if self.radius <= 0:
			raise ValueError("El radio debe ser positivo")

	def contains(self, point: tuple[float, float]) -> bool:
		"""Determina si un punto está dentro del círculo.

		Args:
			point: Punto a evaluar ``(x, y)``.

		Returns:
			True si está dentro o en el borde.
		"""
		dx = point[0] - self.center[0]
		dy = point[1] - self.center[1]
		return dx * dx + dy * dy <= self.radius * self.radius

	def mask(self, shape: tuple[int, int]) -> np.ndarray:
		"""Genera una máscara binaria del círculo.

		Args:
			shape: Dimensiones de la imagen ``(alto, ancho)``.

		Returns:
			Máscara binaria uint8.
		"""
		mask = np.zeros(shape, dtype=np.uint8)
		center_int = tuple(map(int, self.center))
		cv2.circle(mask, center_int, int(self.radius), 255, -1)
		return mask

	def draw(
		self,
		frame: np.ndarray,
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
	) -> None:
		"""Dibuja el círculo sobre un frame.

		Args:
			frame: Imagen destino.
			color: Color del contorno en formato ``(B, G, R)``.
			thickness: Grosor en píxeles.
		"""
		center_int = tuple(map(int, self.center))
		cv2.circle(frame, center_int, int(self.radius), color, thickness)


class CircularFractionRegion(Region):
	"""Región definida por una fracción angular de un círculo (sector circular).

	Attributes:
		region_id: Identificador único de la región.
		center: Centro del sector como ``(x, y)``.
		radius: Radio en píxeles.
		angle_start: Ángulo inicial del sector en grados.
		angle_end: Ángulo final del sector en grados.
		overlap_threshold: Heredado de ``Region``.
	"""

	def __init__(
		self,
		region_id,
		center,
		radius,
		angle_start: float = 0.0,
		angle_end: float = None,
		fraction: float = None,
		overlap_threshold: float = 0.0,
	):
		"""Inicializa una región tipo sector circular.

		Se debe especificar exactamente uno de ``angle_end`` o ``fraction``.

		Args:
			region_id: Identificador único.
			center: Centro del círculo como ``(x, y)``.
			radius: Radio en píxeles.
			angle_start: Ángulo inicial en grados (default: 0.0).
			angle_end: Ángulo final en grados. Excluyente con ``fraction``.
			fraction: Fracción del círculo en ``(0, 1]``. Excluyente con
				``angle_end``. Si se usa, ``angle_end`` se calcula como
				``angle_start + 360 * fraction``.
			overlap_threshold: Fracción mínima de solapamiento. Ver ``Region``.

		Raises:
			ValueError: Si los parámetros son inválidos o overlap_threshold inválido.
		"""
		super().__init__(overlap_threshold)
		self.region_id = region_id
		self.center = (float(center[0]), float(center[1]))
		self.radius = float(radius)

		if self.radius <= 0:
			raise ValueError("El radio debe ser positivo")

		angle_start = float(angle_start) % 360

		if fraction is not None:
			if not (0 < fraction <= 1):
				raise ValueError("fraction debe estar en (0, 1]")
			angle_end = angle_start + 360.0 * float(fraction)

		if angle_end is None:
			raise ValueError("Debes definir angle_end o fraction")

		self.angle_start = angle_start % 360
		self.angle_end = float(angle_end)

	def _angle_in_sector(self, angle: float) -> bool:
		"""Determina si un ángulo en grados cae dentro del sector.

		Maneja correctamente los sectores que cruzan el límite 0°/360°.

		Args:
			angle: Ángulo a evaluar en grados, en ``[0, 360)``.

		Returns:
			True si el ángulo está dentro del sector.
		"""
		if self.angle_start <= self.angle_end:
			return self.angle_start <= angle <= self.angle_end
		else:
			return angle >= self.angle_start or angle <= self.angle_end

	def contains(self, point: tuple[float, float]) -> bool:
		"""Verifica si un punto está dentro del sector circular.

		Args:
			point: Punto a evaluar ``(x, y)``.

		Returns:
			True si el punto pertenece al sector.
		"""
		dx = point[0] - self.center[0]
		dy = point[1] - self.center[1]
		if dx * dx + dy * dy > self.radius * self.radius:
			return False
		angle = math.degrees(math.atan2(dy, dx)) % 360
		return self._angle_in_sector(angle)

	def mask(self, shape: tuple[int, int]) -> np.ndarray:
		"""Genera una máscara binaria del sector circular.

		Maneja correctamente sectores cuyo ``angle_end`` supera los 360°
		dibujando dos arcos parciales cuando el sector cruza el limite 0°/360°.

		Args:
			shape: Dimensiones de la imagen ``(alto, ancho)``.

		Returns:
			Máscara binaria uint8.
		"""
		mask = np.zeros(shape, dtype=np.uint8)
		center_int = tuple(map(int, self.center))
		axes = (int(self.radius), int(self.radius))

		if self.angle_end <= 360:
			# Caso normal: el sector no cruza el limite 0/360
			cv2.ellipse(mask, center_int, axes, 0, self.angle_start, self.angle_end, 255, -1)
		else:
			# El sector cruza el limite: dibujar dos arcos
			cv2.ellipse(mask, center_int, axes, 0, self.angle_start, 360, 255, -1)
			cv2.ellipse(mask, center_int, axes, 0, 0, self.angle_end - 360, 255, -1)

		return mask

	def draw(
		self,
		frame: np.ndarray,
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
	) -> None:
		"""Dibuja el sector circular sobre un frame.

		Args:
			frame: Imagen destino.
			color: Color en formato ``(B, G, R)``.
			thickness: Grosor en píxeles.
		"""
		center_x, center_y = self.center
		cx, cy = int(round(center_x)), int(round(center_y))

		steps = 100
		a_start = math.radians(self.angle_start)
		a_end = math.radians(self.angle_end)

		arc_points = []
		for i in range(steps + 1):
			t = a_start + (a_end - a_start) * i / steps
			x = int(round(center_x + self.radius * math.cos(t)))
			y = int(round(center_y + self.radius * math.sin(t)))
			arc_points.append((x, y))

		points = np.array([(cx, cy)] + arc_points + [(cx, cy)], dtype=np.int32)
		pts = points.reshape((-1, 1, 2))
		cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


class RegionManager:
	"""Contenedor de múltiples regiones de interés.

	Attributes:
		regions: Lista de objetos ``Region``.
	"""

	def __init__(self, regions: list):
		"""Inicializa el gestor de regiones.

		Args:
			regions: Lista de instancias de ``Region``.
		"""
		self.regions = regions