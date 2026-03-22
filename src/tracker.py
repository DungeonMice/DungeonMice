import cv2
import numpy as np

class MouseTracker:
	"""
	Detector de posición del ratón basado en sustracción de fondo.
	Esta clase se encarga exclusivamente de localizar la posición del
	objeto (ratón) en un frame en escala de grises usando técnicas
	clásicas de visión por computadora.
	"""

	def __init__(self, min_area=4000, kernel_size=5, blur_size=0, max_jump=100):
		"""
		Inicializa el detector.

		Parámetros
		----------
		min_area : int
			Área mínima (en píxeles) que debe tener un contorno para
			ser considerado como el ratón. Sirve para filtrar ruido.
		"""
		self.bg = cv2.bgsegm.createBackgroundSubtractorMOG()
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		self.min_area = min_area
		self.blur_size = blur_size  # 0 = sin blur
		self.max_jump = max_jump    # distancia máxima permitida entre detecciones consecutivas
		self.trajectory = []
		self.recording = False
	
	def start_recording(self):
		"""
		Activa la grabación de la trayectoria.
		
		A partir de este momento, cada posición detectada se guardará
		en self.trajectory.
		"""
		self.recording = True
		
	def locate(self, gray_frame):
		"""
		Localiza la posición del ratón en un frame.

		Aplica sustracción de fondo, filtrado morfológico y detección
		de contornos para estimar la posición del objeto.
		Incluye un filtro de salto máximo para ignorar detecciones
		sospechosas que estén muy lejos de la posición anterior.

		Parámetros
		----------
		gray_frame : np.ndarray
			Frame en escala de grises.

		Retorna
		-------
		center_real : tuple or None
			Coordenadas (x, y) del centro del ratón según el contorno
			detectado. Útil para dibujar la hitbox en tiempo real.

		fgmask : np.ndarray
			Máscara binaria resultante de la sustracción de fondo,
			útil para depuración o visualización.
		"""
		# Aplicar blur al frame original antes de la sustracción de fondo
		# Reduce ruido de textura y movimiento sutil de cámara
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

		cx_real = int(M["m10"]/M["m00"])
		cy_real = int(M["m01"]/M["m00"])

		# Filtro de salto máximo: ignorar si el ratón "saltó" demasiado lejos
		if self.recording and len(self.trajectory) > 0:
			last_pos = self.trajectory[-1]
			dx = cx_real - last_pos[0]
			dy = cy_real - last_pos[1]
			dist = np.sqrt(dx*dx + dy*dy)
			if dist > self.max_jump:
				return None, fgmask  # detección sospechosa, ignorar

		# Solo guardar posición si estamos en modo grabación
		if self.recording:
			center_real = (cx_real, cy_real)
			self.trajectory.append(center_real)
			return center_real, fgmask
		return None, fgmask