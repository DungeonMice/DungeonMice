import cv2
import numpy as np


class MouseTracker:
	"""Detector de posición del ratón basado en sustracción de fondo.

	Se encarga exclusivamente de localizar la posición del objeto (ratón)
	en un frame en escala de grises usando técnicas clásicas de visión
	por computadora.

	Attributes:
		bg: Sustractor de fondo MOG de OpenCV.
		kernel: Elemento estructurante elíptico para operaciones morfológicas.
		min_area: Área mínima en píxeles para considerar un contorno como el ratón.
		blur_size: Tamaño del kernel de GaussianBlur. 0 desactiva el blur.
		max_jump: Distancia máxima permitida en píxeles entre detecciones consecutivas.
		trajectory: Lista de posiciones (x, y) grabadas desde que se activó la grabación.
		recording: True si la grabación de trayectoria está activa.
	"""

	def __init__(self, min_area=4000, kernel_size=5, blur_size=0, max_jump=100):
		"""Inicializa el detector de posición del ratón.

		Args:
			min_area: Área mínima en píxeles que debe tener un contorno para
				ser considerado como el ratón. Sirve para filtrar ruido.
			kernel_size: Tamaño del elemento estructurante elíptico usado en
				las operaciones morfológicas.
			blur_size: Tamaño del kernel de GaussianBlur aplicado antes de la
				sustracción de fondo. 0 desactiva el blur.
			max_jump: Distancia máxima en píxeles permitida entre dos detecciones
				consecutivas. Detecciones que superen este umbral se descartan
				como falsas.
		"""
		self.bg = cv2.bgsegm.createBackgroundSubtractorMOG()
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		self.min_area = min_area
		self.blur_size = blur_size
		self.max_jump = max_jump
		self.trajectory = []
		self.recording = False

	def start_recording(self):
		"""Activa la grabación de la trayectoria.

		A partir de este momento, cada posición detectada se añade
		a ``self.trajectory``.
		"""
		self.recording = True

	def locate(self, gray_frame):
		"""Localiza la posición del ratón en un frame en escala de grises.

		Aplica sustracción de fondo, filtrado morfológico y detección de
		contornos para estimar la posición del objeto. Incluye un filtro
		de salto máximo para descartar detecciones sospechosas muy alejadas
		de la posición anterior.

		Args:
			gray_frame: Frame en escala de grises (np.ndarray).

		Returns:
			Tupla ``(center, fgmask)`` donde:

			- ``center``: Coordenadas ``(x, y)`` del centroide del contorno
			  detectado, o ``None`` si no se encontró ninguna detección válida.
			- ``fgmask``: Máscara binaria (np.ndarray) resultante de la
			  sustracción de fondo, útil para depuración y visualización.
		"""
		# Aplicar blur antes de la sustracción de fondo para reducir ruido
		# de textura y movimiento sutil de cámara
		if self.blur_size > 0:
			gray_frame = cv2.GaussianBlur(gray_frame, (self.blur_size, self.blur_size), 0)

		fgmask = self.bg.apply(gray_frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)  # rellena huecos dentro del ratón
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)   # elimina ruido pequeño
		fgmask = cv2.dilate(fgmask, self.kernel, iterations=2)            # une partes separadas del ratón

		cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		if not cnts:
			return None, fgmask

		cnt = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(cnt) < self.min_area:
			return None, fgmask

		# Centroide del contorno
		M = cv2.moments(cnt)
		if M["m00"] == 0:
			return None, fgmask

		cx_real = int(M["m10"] / M["m00"])
		cy_real = int(M["m01"] / M["m00"])

		# Filtro de salto máximo: descartar si el ratón "saltó" demasiado lejos
		if self.recording and len(self.trajectory) > 0:
			last_pos = self.trajectory[-1]
			dx = cx_real - last_pos[0]
			dy = cy_real - last_pos[1]
			dist = np.sqrt(dx * dx + dy * dy)
			if dist > self.max_jump:
				return None, fgmask

		# Solo guardar posición si la grabación está activa
		if self.recording:
			center_real = (cx_real, cy_real)
			self.trajectory.append(center_real)
			return center_real, fgmask

		return None, fgmask

	def get_total_distance(self) -> float:
		"""Calcula la distancia total recorrida a lo largo de la trayectoria grabada.

		Suma las distancias euclidianas entre cada par de puntos consecutivos
		en ``self.trajectory``.

		Returns:
			Distancia total en píxeles. Retorna 0.0 si hay menos de 2 puntos.
		"""
		if len(self.trajectory) < 2:
			return 0.0
		total = 0.0
		for i in range(1, len(self.trajectory)):
			dx = self.trajectory[i][0] - self.trajectory[i - 1][0]
			dy = self.trajectory[i][1] - self.trajectory[i - 1][1]
			total += (dx * dx + dy * dy) ** 0.5
		return total