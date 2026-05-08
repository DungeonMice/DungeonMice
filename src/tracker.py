import cv2
import numpy as np

# Ventana de frames al inicio de la grabación en que se monitorea la
# estabilidad de CSRT. Si hay más de _CSRT_MAX_REINITS re-inicializaciones
# en ese período, CSRT se desactiva para ese video.
_CSRT_CHECK_WINDOW = 80
_CSRT_MAX_REINITS  = 6


class MouseTracker:
	"""Detector híbrido MOG2 + CSRT para localizar la posición del ratón.

	Combina dos estrategias complementarias:

	- **MOG2** (fondo): detecta el ratón como el blob de primer plano más
	  grande dentro del área válida. Robusto para re-adquirir al ratón desde
	  cero, pero sensible a cambios bruscos de iluminación.
	- **CSRT** (objeto): tracker discriminativo que sigue al ratón frame a
	  frame usando características visuales. No depende del modelo de fondo,
	  por lo que es robusto a flashes de luz, pero puede derivar si el ratón
	  se queda quieto mucho tiempo.

	Lógica de combinación por frame:
	  1. MOG2 intenta detectar un blob válido (área, máscara, filtro de salto).
	  2. CSRT actualiza su posición.
	  3. Si MOG2 detecta → posición MOG2 es la definitiva. Si además CSRT
	     ha derivado más de ``csrt_reinit_dist`` píxeles, se re-inicializa.
	  4. Si MOG2 falla pero CSRT sigue → posición CSRT como fallback.
	  5. Si ambos fallan → interpola repitiendo la última posición conocida
	     hasta ``max_missing_frames``; después devuelve None.

	CSRT se inicializa en la primera detección válida de MOG2 (puede ocurrir
	durante el warmup) y usa el frame gris sin blur para preservar texturas.

	Attributes:
		bg (cv2.BackgroundSubtractorMOG2): Sustractor de fondo MOG2.
		kernel (np.ndarray): Kernel elíptico para open y dilate.
		_close_kernel (np.ndarray): Kernel fijo 5×5 para el close inicial.
		min_area (int): Área mínima en píxeles para contornos válidos.
		blur_size (int): Tamaño del GaussianBlur para MOG2. 0 = sin blur.
		max_jump (int): Distancia máxima entre detecciones consecutivas
			durante la grabación.
		max_missing_frames (int): Frames sin detección antes de devolver None.
		recording_lr (float): Learning rate del MOG2 durante la grabación.
		use_csrt (bool): Si False desactiva CSRT completamente (solo MOG2).
		csrt_reinit_dist (int): Distancia en píxeles entre MOG2 y CSRT a
			partir de la cual se re-inicializa CSRT.
		trajectory (list[tuple[int,int]]): Posiciones (x, y) grabadas.
		recording (bool): True si la grabación está activa.
		valid_mask (np.ndarray | None): Máscara binaria del laberinto.
		confirm_dist (int): Umbral de distancia (px) para clasificar un
			movimiento como "grande" y someterlo al filtro de confirmación.
			Se aplica tanto a posiciones de la trayectoria como a la decisión
			de re-inicializar CSRT desde MOG2.
		min_confirm_frames (int): Frames consecutivos requeridos para aceptar
			un movimiento grande o una re-inicialización de CSRT. Con el
			valor por defecto (2), un falso positivo de 1 frame queda
			silenciado sin introducir retardo perceptible en el tracking real.
	"""

	def __init__(
		self,
		min_area: int = 400,
		kernel_size: int = 5,
		blur_size: int = 0,
		max_jump: int = 100,
		max_missing_frames: int = 5,
		mog_history: int = 500,
		mog_threshold: int = 30,
		recording_lr: float = 0.002,
		use_csrt: bool = True,
		csrt_reinit_dist: int = 60,
		confirm_dist: int = 50,
		min_confirm_frames: int = 2,
	):
		"""Inicializa el tracker híbrido.

		Args:
			min_area: Área mínima en píxeles para considerar un contorno como
				el ratón. 400px² es suficientemente bajo para detectar ratones
				delgados; el filtro de confirmación y max_jump se encargan de
				rechazar el ruido pequeño que pueda pasar.
			kernel_size: Tamaño del elemento estructurante elíptico para open y dilate.
			blur_size: Tamaño del GaussianBlur aplicado antes de MOG2. 0 = sin blur.
				CSRT siempre recibe el frame sin blur.
			max_jump: Distancia máxima en píxeles entre detecciones consecutivas
				durante la grabación. Rechaza saltos que indican falso positivo.
			max_missing_frames: Frames sin detección antes de devolver None.
			mog_history: Frames que MOG2 usa para construir el modelo de fondo.
			mog_threshold: Umbral de varianza de MOG2 (varThreshold).
			recording_lr: Learning rate del MOG2 durante la grabación.
			use_csrt: Si True activa el fallback CSRT. Si False usa solo MOG2.
			csrt_reinit_dist: Distancia (px) entre MOG2 y CSRT que dispara
				una re-inicialización del tracker.
			confirm_dist: Movimientos más grandes que este valor (px) requieren
				confirmación en frames consecutivos antes de aceptarse.
			min_confirm_frames: Frames consecutivos requeridos para confirmar
				un movimiento grande o re-init de CSRT.
		"""
		self.bg = cv2.createBackgroundSubtractorMOG2(
			history=mog_history,
			varThreshold=mog_threshold,
			detectShadows=False,
		)
		self.kernel           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		self._close_kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		self.min_area         = min_area
		self.blur_size        = blur_size
		self.max_jump         = max_jump
		self.max_missing_frames = max_missing_frames
		self.recording_lr     = recording_lr
		self.use_csrt         = use_csrt
		self.csrt_reinit_dist = csrt_reinit_dist
		self.confirm_dist     = confirm_dist
		self.min_confirm_frames = min_confirm_frames

		self.trajectory: list                   = []
		self.recording: bool                    = False
		self._last_known_position: tuple | None = None
		self._missing_frames: int               = 0

		self.valid_mask: np.ndarray | None  = None
		self._border_mask: np.ndarray | None = None

		# CSRT solo activo durante grabación (no en warmup).
		self._csrt                   = None
		self._csrt_initialized: bool = False

		# Contadores para el auto-disable de CSRT (umbrales en constantes de módulo).
		self._csrt_reinit_count: int = 0
		self._csrt_check_frames: int = 0

		# Filtro de confirmación para posiciones de la trayectoria.
		self._candidate_pos: tuple | None = None
		self._candidate_frames: int       = 0

		# Filtro de confirmación para re-inicialización de CSRT.
		self._reinit_candidate_pos: tuple | None = None
		self._reinit_candidate_frames: int       = 0

	# ------------------------------------------------------------------
	# Configuración
	# ------------------------------------------------------------------

	def set_valid_mask(self, mask: np.ndarray | None) -> None:
		"""Define la máscara binaria del área válida del laberinto.

		Args:
			mask (np.ndarray | None): Array uint8 con 255 en zonas válidas y
				0 fuera. None desactiva el filtro de máscara (se usa solo
				un margen de borde mínimo).
		"""
		self.valid_mask = mask
		if mask is not None:
			print(f"[Tracker] valid_mask — shape={mask.shape}, "
				  f"píxeles válidos={cv2.countNonZero(mask)}")
		else:
			print("[Tracker] valid_mask=None — se usará margen de borde por defecto")

	def _get_active_mask(self, h: int, w: int) -> np.ndarray:
		"""Retorna la máscara activa para el frame actual.

		Si ``valid_mask`` está definida la usa directamente. Si no, construye
		una máscara con un margen de 20 px en cada borde para descartar
		artefactos de los bordes de la imagen.

		Args:
			h (int): Alto del frame en píxeles.
			w (int): Ancho del frame en píxeles.

		Returns:
			np.ndarray: Array uint8 con 255 en la zona válida.
		"""
		if self.valid_mask is not None:
			return self.valid_mask

		if self._border_mask is None or self._border_mask.shape != (h, w):
			m = 20
			self._border_mask = np.zeros((h, w), dtype=np.uint8)
			self._border_mask[m:h - m, m:w - m] = 255
		return self._border_mask

	# ------------------------------------------------------------------
	# Control de grabación
	# ------------------------------------------------------------------

	def start_recording(self) -> None:
		"""Activa la grabación y resetea todos los buffers de estado."""
		self.recording              = True
		self._missing_frames        = 0
		self._csrt_reinit_count     = 0
		self._csrt_check_frames     = 0
		self._candidate_pos         = None
		self._candidate_frames      = 0
		self._reinit_candidate_pos    = None
		self._reinit_candidate_frames = 0

	# ------------------------------------------------------------------
	# Detección MOG2
	# ------------------------------------------------------------------

	def _mog2_detect(
		self,
		blur_frame: np.ndarray,
		active_mask: np.ndarray,
	) -> tuple:
		"""Ejecuta el pipeline MOG2 y devuelve el mejor candidato.

		Pipeline:
		  1. MOG2 con learning rate adaptado a la fase (warmup vs grabación).
		  2. Enmascara con ``active_mask`` para ignorar zonas fuera del laberinto.
		  3. Morfología: close 5×5 → open con kernel usuario → dilate ×2.
		  4. Entre los contornos válidos (área ≥ min_area, centroide en máscara)
		     elige el más cercano a ``_last_known_position``, o el más grande
		     si aún no hay posición de referencia.

		Args:
			blur_frame (np.ndarray): Frame en escala de grises, posiblemente
				con GaussianBlur aplicado.
			active_mask (np.ndarray): Máscara binaria de la zona válida.

		Returns:
			tuple: ``(pos, bbox, fgmask)`` donde:
			  - ``pos``: ``(cx, cy)`` del mejor contorno, o None.
			  - ``bbox``: ``(x, y, w, h)`` bounding rect del contorno, o None.
			  - ``fgmask``: máscara binaria procesada del frame.
		"""
		lr = self.recording_lr if self.recording else -1
		fgmask = self.bg.apply(blur_frame, learningRate=lr)

		fgmask = cv2.bitwise_and(fgmask, active_mask)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self._close_kernel)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  self.kernel)
		fgmask = cv2.dilate(fgmask, self.kernel, iterations=2)

		cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		if not cnts:
			return None, None, fgmask

		# Filtrar candidatos por área y por centroide dentro de la máscara
		candidates = []
		for c in cnts:
			if cv2.contourArea(c) < self.min_area:
				continue
			M = cv2.moments(c)
			if M["m00"] == 0:
				continue
			cx = int(M["m10"] / M["m00"])
			cy = int(M["m01"] / M["m00"])
			if active_mask[cy, cx] == 0:
				continue
			candidates.append((cx, cy, cv2.contourArea(c), c))

		if not candidates:
			return None, None, fgmask

		# Selección: más cercano a _last_known_position, o más grande
		if self._last_known_position is not None:
			lx, ly = self._last_known_position
			best = min(candidates, key=lambda c: (c[0] - lx) ** 2 + (c[1] - ly) ** 2)
		else:
			best = max(candidates, key=lambda c: c[2])

		cx, cy, _, cnt = best
		x, y, w, h = cv2.boundingRect(cnt)
		return (cx, cy), (x, y, w, h), fgmask

	# ------------------------------------------------------------------
	# CSRT
	# ------------------------------------------------------------------

	def _init_csrt(self, gray_frame: np.ndarray, bbox: tuple) -> None:
		"""Inicializa o re-inicializa el tracker CSRT con un bounding box.

		Si durante los primeros _CSRT_CHECK_WINDOW frames de grabación el número
		de re-inicializaciones supera _CSRT_MAX_REINITS, CSRT se auto-desactiva:
		indica que el video no es estable y CSRT está causando más daño que bien.

		Args:
			gray_frame: Frame gris sin blur.
			bbox: ``(x, y, w, h)`` del objeto a seguir.
		"""
		# Contar re-inits en ventana de estabilidad (solo durante grabación)
		if self.recording and self._csrt_check_frames < _CSRT_CHECK_WINDOW:
			self._csrt_reinit_count += 1
			if self._csrt_reinit_count > _CSRT_MAX_REINITS:
				self.use_csrt = False
				self._csrt_initialized = False
				print(f"[CSRT] Auto-desactivado: {self._csrt_reinit_count} re-inits "
					  f"en primeros {self._csrt_check_frames} frames — usando solo MOG2")
				return

		self._csrt = cv2.TrackerCSRT_create() # type: ignore
		self._csrt.init(gray_frame, bbox)
		self._csrt_initialized = True
		# print(f"[CSRT] init bbox={bbox}")  # descomentar para diagnóstico

	def _update_csrt(self, gray_frame: np.ndarray) -> tuple | None:
		"""Actualiza el tracker CSRT y devuelve la posición estimada.

		Args:
			gray_frame (np.ndarray): Frame gris sin blur del instante actual.

		Returns:
			tuple | None: ``(cx, cy)`` del centro del bounding box, o None si
			CSRT falló o no está inicializado.
		"""
		if not self._csrt_initialized or self._csrt is None:
			return None

		success, bbox = self._csrt.update(gray_frame)
		if not success:
			self._csrt_initialized = False
			return None

		x, y, w, h = [int(v) for v in bbox]
		cx, cy = x + w // 2, y + h // 2

		# Si CSRT derivó fuera del área válida del laberinto, invalidarlo.
		# MOG2 ya respeta valid_mask, pero CSRT no tiene esa restricción y puede
		# seguir objetos fuera del laberinto produciendo trayectorias falsas.
		if self.valid_mask is not None:
			mh, mw = self.valid_mask.shape[:2]
			if not (0 <= cy < mh and 0 <= cx < mw and self.valid_mask[cy, cx] > 0):
				self._csrt_initialized = False
				return None

		return (cx, cy)

	# ------------------------------------------------------------------
	# Filtro de confirmación
	# ------------------------------------------------------------------

	def _apply_confirmation(self, pos: tuple) -> tuple | None:
		"""Filtra detecciones falsas puntuales mediante confirmación multi-frame.

		Si la nueva posición implica un movimiento grande (>= ``confirm_dist``
		respecto a la última posición aceptada), se bufferiza como candidata.
		Solo se acepta cuando la detección sigue en el mismo vecindario durante
		``min_confirm_frames`` frames consecutivos.

		Movimientos pequeños (<``confirm_dist``) se aceptan directamente, lo que
		garantiza que el tracking normal continuo no introduce ningún retardo.

		Args:
			pos (tuple): Posición ``(x, y)`` detectada en el frame actual.

		Returns:
			tuple | None: La posición confirmada para añadir a la trayectoria,
			o None si la posición está aún pendiente de confirmación.
		"""
		if self._last_known_position is None:
			# Sin ancla — aceptar directamente (primer frame grabado)
			self._candidate_pos    = None
			self._candidate_frames = 0
			return pos

		dist = np.hypot(
			pos[0] - self._last_known_position[0],
			pos[1] - self._last_known_position[1],
		)

		if dist < self.confirm_dist:
			# Movimiento pequeño — aceptar de inmediato, limpiar candidato pendiente
			self._candidate_pos    = None
			self._candidate_frames = 0
			return pos

		# Movimiento grande — requiere confirmación
		if self._candidate_pos is not None:
			candidate_dist = np.hypot(
				pos[0] - self._candidate_pos[0],
				pos[1] - self._candidate_pos[1],
			)
			if candidate_dist < self.confirm_dist:
				# El detector sigue apuntando al mismo candidato → confirmar
				self._candidate_frames += 1
				self._candidate_pos = pos  # actualizar al centroide más reciente
				if self._candidate_frames >= self.min_confirm_frames:
					# Confirmado: aceptar y limpiar buffer
					self._candidate_pos    = None
					self._candidate_frames = 0
					return pos
				# Aún no suficientes frames de confirmación
				return None
			# El candidato cambió de lugar: reiniciar buffer con la nueva posición
		self._candidate_pos    = pos
		self._candidate_frames = 1
		return None  # Esperando confirmación

	# ------------------------------------------------------------------
	# Loop principal
	# ------------------------------------------------------------------

	def locate(self, gray_frame: np.ndarray) -> tuple:
		"""Localiza la posición del ratón en un frame en escala de grises.

		Ejecuta el pipeline híbrido MOG2 + CSRT descrito en la clase.

		Args:
			gray_frame (np.ndarray): Frame en escala de grises.

		Returns:
			tuple[tuple[int,int] | None, np.ndarray]: Par ``(pos, fgmask)``
			donde ``pos`` es ``(x, y)`` o None, y ``fgmask`` es la máscara
			binaria de MOG2 procesada (útil para visualización).
		"""
		h, w = gray_frame.shape[:2]
		active_mask = self._get_active_mask(h, w)

		# Frame para MOG2 (con blur opcional) y para CSRT (sin blur siempre)
		blur_frame = (
			cv2.GaussianBlur(gray_frame, (self.blur_size, self.blur_size), 0)
			if self.blur_size > 0
			else gray_frame
		)

		# 1. Detección MOG2
		mog2_pos, mog2_bbox, fgmask = self._mog2_detect(blur_frame, active_mask)

		# 2. Actualización CSRT — solo durante grabación.
		#    Incrementar contador para la ventana de estabilidad.
		if self.recording:
			self._csrt_check_frames += 1

		csrt_pos = (
			self._update_csrt(gray_frame)
			if (self.use_csrt and self.recording)
			else None
		)

		# 3. Posición final: CSRT primario (suave), MOG2 para corrección.
		#
		# Prioridad de decisión:
		#   a) Ambos disponibles y de acuerdo  → CSRT (tracking continuo, sin jitter).
		#   b) Ambos disponibles y divergen    → aplicar confirmación antes de re-init
		#                                        CSRT: si MOG2 lleva N frames apuntando
		#                                        al mismo sitio se acepta; si no, se
		#                                        mantiene CSRT para evitar artefactos.
		#   c) Solo CSRT                        → CSRT (MOG2 perdido por ruido).
		#   d) Solo MOG2                        → MOG2 e inicializa CSRT (solo si graba).
		#   e) Ninguno                          → interpolar.
		#
		# La confirmación antes del re-init previene el artefacto de "teleportación":
		# un falso positivo puntual de MOG2 no re-inicializa CSRT en esa posición
		# ni crea una línea diagonal en la trayectoria.
		final_pos = None

		if csrt_pos is not None and mog2_pos is not None:
			drift = np.hypot(
				mog2_pos[0] - csrt_pos[0],
				mog2_pos[1] - csrt_pos[1],
			)
			if drift > self.csrt_reinit_dist:
				# Divergencia grande: ¿re-adquisición real o falso positivo de MOG2?
				# Bufferear la posición de MOG2 y solo aceptarla cuando se confirme
				# en min_confirm_frames frames consecutivos.
				if (
					self._reinit_candidate_pos is not None
					and np.hypot(
						mog2_pos[0] - self._reinit_candidate_pos[0],
						mog2_pos[1] - self._reinit_candidate_pos[1],
					) < self.confirm_dist
				):
					self._reinit_candidate_frames += 1
					self._reinit_candidate_pos = mog2_pos  # actualizar al centroide más reciente
				else:
					# Nuevo candidato (o el candidato anterior cambió de lugar)
					self._reinit_candidate_pos    = mog2_pos
					self._reinit_candidate_frames = 1

				if self._reinit_candidate_frames >= self.min_confirm_frames:
					# Confirmado: re-inicializar CSRT y confiar en MOG2
					if mog2_bbox is not None:
						self._init_csrt(gray_frame, mog2_bbox)
					self._reinit_candidate_pos    = None
					self._reinit_candidate_frames = 0
					final_pos = mog2_pos
				else:
					# Aún no confirmado: mantener CSRT para evitar la línea diagonal
					final_pos = csrt_pos
			else:
				# Acuerdo entre CSRT y MOG2: CSRT es más suave, usarlo
				self._reinit_candidate_pos    = None
				self._reinit_candidate_frames = 0
				final_pos = csrt_pos

		elif csrt_pos is not None:
			# MOG2 falló (ruido de fondo); CSRT sigue al ratón sin problema
			self._reinit_candidate_pos    = None
			self._reinit_candidate_frames = 0
			final_pos = csrt_pos

		elif mog2_pos is not None:
			# CSRT no está listo o se perdió; usar MOG2 e inicializar CSRT.
			# CSRT solo se inicializa durante la grabación (no en el warmup).
			final_pos = mog2_pos
			if self.use_csrt and mog2_bbox is not None and not self._csrt_initialized and self.recording:
				self._init_csrt(gray_frame, mog2_bbox)

		if final_pos is None:
			return self._interpolate(fgmask)

		# 5. Filtro de salto — solo durante grabación, basado en trayectoria.
		#    Rechaza teleportaciones grandes (>max_jump) que indican falso positivo
		#    lejano o bug del CSRT. El filtro de confirmación (paso 6) cubre los
		#    falsos positivos cercanos que este filtro no alcanza.
		if self.recording and self.trajectory:
			last = self.trajectory[-1]
			if np.hypot(final_pos[0] - last[0], final_pos[1] - last[1]) > self.max_jump:
				return self._interpolate(fgmask)

		# 6. Actualizar estado
		if self.recording:
			# Filtro de confirmación: movimientos grandes se bufferean hasta
			# que se confirmen en min_confirm_frames frames consecutivos.
			# Movimientos pequeños (<confirm_dist) se aceptan de inmediato.
			confirmed = self._apply_confirmation(final_pos)
			if confirmed is None:
				# Posición aún pendiente de confirmación: repetir última conocida
				# para no introducir un punto falso en la trayectoria.
				return self._interpolate(fgmask)
			self.trajectory.append(confirmed)
			self._last_known_position = confirmed
			self._missing_frames = 0
			return confirmed, fgmask

		# Durante warmup: actualizar ancla para que MOG2 seleccione por
		# proximidad desde el primer frame grabado, pero sin tocar la trayectoria.
		self._last_known_position = final_pos
		return None, fgmask

	# ------------------------------------------------------------------
	# Interpolación
	# ------------------------------------------------------------------

	def _interpolate(self, fgmask: np.ndarray) -> tuple:
		"""Repite la última posición conocida hasta ``max_missing_frames``.

		Pasado ese límite devuelve None para indicar pérdida total.
		``_last_known_position`` nunca se borra: sigue siendo el ancla para
		el filtro de salto cuando el ratón sea re-detectado.

		Args:
			fgmask (np.ndarray): Máscara binaria del frame actual.

		Returns:
			tuple: ``(_last_known_position, fgmask)`` si hay margen de
			interpolación, ``(None, fgmask)`` si se superó el límite o no
			hay posición conocida.
		"""
		if self.recording and self._last_known_position is not None:
			self._missing_frames += 1
			if self._missing_frames <= self.max_missing_frames:
				self.trajectory.append(self._last_known_position)
				return self._last_known_position, fgmask
		return None, fgmask

	# ------------------------------------------------------------------
	# Métricas
	# ------------------------------------------------------------------

	def get_total_distance(self) -> float:
		"""Calcula la distancia total recorrida en la trayectoria grabada.

		Returns:
			float: Distancia total en píxeles. 0.0 si hay menos de 2 puntos.
		"""
		if len(self.trajectory) < 2:
			return 0.0
		total = 0.0
		for i in range(1, len(self.trajectory)):
			dx = self.trajectory[i][0] - self.trajectory[i - 1][0]
			dy = self.trajectory[i][1] - self.trajectory[i - 1][1]
			total += (dx * dx + dy * dy) ** 0.5
		return total
