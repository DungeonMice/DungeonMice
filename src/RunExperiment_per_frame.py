"""frame_engine.py
==================

Motor de análisis frame a frame controlado externamente.

El analizador no avanza por su cuenta — procesa exactamente un frame
por cada llamada a ``next()``, y queda suspendido entre llamadas.
No se gasta ningún recurso si no se solicita avance.

Los frames ya procesados se almacenan en un cache interno indexado,
lo que permite a la GUI acceder a cualquier frame anterior sin
reanalizar nada.

Componentes
-----------
FrameData
    Resultado de procesar un frame: imagen renderizada + datos de análisis.
VideoAnalyzer
    Procesa un frame por llamada a ``next()``. Mantiene cache de todos
    los frames procesados. No corre en segundo plano.

Uso típico::

    from frame_engine import VideoAnalyzer

    analyzer = VideoAnalyzer(labyrinth)

    # Avanzar un frame (analiza y devuelve)
    frame_data = analyzer.next()

    # Avanzar varios frames
    for _ in range(10):
        frame_data = analyzer.next()

    # Acceder a un frame ya analizado sin reanalizar
    frame_data = analyzer.get(0)   # primer frame
    frame_data = analyzer.get(-1)  # último frame procesado hasta ahora
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from tracker import MouseTracker
from logic import EventLogic
from visualizer import ExperimentVisualizer


# ---------------------------------------------------------------------------
# Estructura de datos
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Resultado del análisis de un único frame.

    Attributes:
        index: Índice absoluto del frame dentro del video (empieza en 1, igual que RunExperiment).
        time: Tiempo en segundos correspondiente al frame.
        image: Frame renderizado en BGR con regiones, hitbox y trayectoria
            dibujadas. Listo para mostrar en la GUI.
        position: Posición detectada del sujeto como ``(x, y)``, o ``None``
            si no se detectó o si el frame es anterior a ``start_time``.
        region_states: Snapshot del diccionario ``{region_id: ZoneState}``
            con el estado de cada región en este frame exacto.
        recording_active: ``True`` si el frame pertenece al período de
            grabación (posterior a ``start_time``).
    """
    index:            int
    time:             float
    image:            np.ndarray
    position:         tuple[int, int] | None
    region_states:    dict
    recording_active: bool


# ---------------------------------------------------------------------------
# Analizador
# ---------------------------------------------------------------------------

class VideoAnalyzer:
    """Analizador de video controlado externamente, frame a frame.

    Procesa exactamente un frame por llamada a ``next()``. Entre llamadas
    no consume CPU ni memoria adicional. El estado del tracker MOG se
    preserva correctamente porque el análisis siempre es secuencial.

    Los frames procesados se acumulan en un cache interno (``_cache``)
    accesible por índice mediante ``get()``. La GUI puede navegar
    libremente por los frames ya procesados sin ningún costo adicional.

    Attributes:
        total_frames: Número total de frames del video. Disponible tras
            la inicialización.
        fps: FPS del video. Disponible tras la inicialización.
        processed_count: Número de frames procesados hasta el momento.

    Args:
        labyrinth: Objeto ``Labyrinth`` con todos los parámetros del
            experimento.

    Raises:
        RuntimeError: Si el video no se puede abrir al inicializar.

    Example::

        analyzer = VideoAnalyzer(labyrinth)

        while not analyzer.is_done():
            frame_data = analyzer.next()
            gui.display(frame_data.image)
            if user_paused:
                break  # el analizador queda suspendido aquí
    """

    def __init__(self, labyrinth):
        """Inicializa el analizador y abre el video.

        Abre la captura de video y construye el tracker, la lógica de
        eventos y el visualizador. No procesa ningún frame todavía.

        Args:
            labyrinth: Objeto ``Labyrinth`` configurado con ``video_path``,
                ``start_time``, ``min_detection_area``, ``kernel_size``,
                ``blur_size``, ``mog_threshold``, ``recording_lr``,
                ``use_csrt``, ``mog_history`` y ``hitbox_size``.

        Raises:
            RuntimeError: Si el video no se puede abrir.
        """
        self._labyrinth = labyrinth

        self._cap = cv2.VideoCapture(labyrinth.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {labyrinth.video_path}")

        self.fps          = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width             = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height            = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        labyrinth.fps     = self.fps

        self._start_frame = int(labyrinth.start_time * self.fps)
        self._frame_idx   = 0
        self._done        = False
        self._cache: list[FrameData] = []

        valid_mask = labyrinth.get_valid_mask(height, width)

        self._tracker = MouseTracker(
            min_area=labyrinth.min_detection_area,
            kernel_size=labyrinth.kernel_size,
            blur_size=labyrinth.blur_size,
            mog_threshold=labyrinth.mog_threshold,
            recording_lr=labyrinth.recording_lr,
            use_csrt=labyrinth.use_csrt,
            mog_history=labyrinth.mog_history,
        )
        self._tracker.set_valid_mask(valid_mask)

        self._logic = EventLogic(labyrinth.regions,
                                 hitbox_size=labyrinth.hitbox_size)
        self._visualizer = ExperimentVisualizer(labyrinth.regions,
                                                hitbox_size=labyrinth.hitbox_size)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    @property
    def processed_count(self) -> int:
        """Número de frames procesados hasta el momento.

        Returns:
            Entero >= 0.
        """
        return len(self._cache)

    def is_done(self) -> bool:
        """Indica si el video fue procesado en su totalidad.

        Returns:
            ``True`` cuando no quedan más frames por analizar.
        """
        return self._done

    def next(self) -> FrameData | None:
        """Procesa el siguiente frame y lo devuelve.

        Avanza exactamente un frame en el video. Si el video ya fue
        procesado completamente, devuelve ``None`` sin hacer nada.

        Returns:
            El ``FrameData`` del frame recién procesado, o ``None`` si
            el video ya terminó.
        """
        if self._done:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self._done = True
            self._cap.release()
            return None

        frame_data = self._process_frame(frame)
        self._cache.append(frame_data)
        self._frame_idx += 1
        return frame_data

    def get(self, index: int) -> FrameData | None:
        """Devuelve un frame ya procesado del cache por índice.

        No analiza nada — accede directamente al cache. Acepta índices
        negativos (``-1`` = último procesado hasta ahora).

        Args:
            index: Índice del frame. Negativo cuenta desde el último
                disponible.

        Returns:
            El ``FrameData`` correspondiente, o ``None`` si el índice
            está fuera del rango de frames ya procesados.
        """
        if not self._cache:
            return None
        try:
            return self._cache[index]
        except IndexError:
            return None

    def release(self) -> None:
        """Libera el recurso de video explícitamente.

        Se llama automáticamente cuando el video termina. Útil si se
        quiere interrumpir el análisis antes de que el video finalice.
        """
        if not self._done:
            self._cap.release()
            self._done = True

    # ------------------------------------------------------------------
    # Procesamiento interno
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> FrameData:
        """Procesa un frame crudo y produce un FrameData.

        Aplica el tracker, actualiza la lógica de eventos y renderiza
        las anotaciones sobre el frame.

        Args:
            frame: Frame BGR crudo leído del video.

        Returns:
            FrameData con imagen renderizada y snapshot del estado.
        """
        idx              = self._frame_idx + 1   # convención 1-based, igual que RunExperiment
        t                = idx / self.fps
        recording_active = idx >= self._start_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if idx == self._start_frame:
            self._tracker.start_recording()

        pos_real, fgmask = self._tracker.locate(gray)

        if recording_active:
            self._logic.update(pos_real, t, idx)
            self._labyrinth.process_frame(position=pos_real, time=t)

        rendered = frame.copy()
        self._visualizer.draw_regions(rendered, self._logic.states)
        self._visualizer.draw_hitbox(rendered, pos_real, self._logic.states)
        self._visualizer.draw_trajectory_on_mask(
            fgmask, self._tracker.trajectory, idx, self._start_frame
        )

        cv2.putText(rendered, f"Frame {idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(rendered, f"t = {t:.2f} s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return FrameData(
            index=idx,
            time=t,
            image=rendered,
            position=pos_real if recording_active else None,
            region_states=copy.deepcopy(self._logic.states),
            recording_active=recording_active,
        )
