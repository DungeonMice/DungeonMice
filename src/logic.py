"""logic.py
=========

Lógica de eventos de entrada y salida en regiones de interés.

Este módulo es independiente de OpenCV y de cualquier interfaz gráfica.
Recibe posiciones y tiempos, y mantiene el historial de cada región.
"""


class ZoneState:
    """Estado temporal asociado a una región de interés.

    Esta clase no conoce geometría ni video. Únicamente almacena
    el historial lógico de una región: si el objeto está dentro,
    cuándo entró, cuántas veces entró y cuánto tiempo total permaneció.

    Attributes:
        inside: True si el objeto está actualmente dentro de la región.
        enter_time: Timestamp en segundos del último ingreso, o None si
            el objeto no está dentro.
        enter_frame: Lista de frames absolutos en los que ocurrió una entrada.
        exit_frame: Lista de frames absolutos en los que ocurrió una salida.
        total_time: Tiempo total acumulado en segundos dentro de la región.
        entries: Número total de entradas a la región.
        events: Lista de tuplas (enter_time, exit_time) en segundos para
            análisis detallado por evento.
    """

    def __init__(self):
        """Inicializa el estado de una región con todos los contadores en cero."""
        self.inside = False
        self.enter_time = None
        self.enter_frame = []
        self.exit_frame = []
        self.total_time = 0
        self.entries = 0
        self.events = []


class EventLogic:
    """Lógica de eventos de entrada y salida para múltiples regiones.

    Recibe posiciones del objeto a lo largo del tiempo y determina
    eventos de entrada y salida en cada región de interés. No realiza
    detección visual ni define geometría: solo coordina estados.

    La detección de pertenencia se delega a ``Region.contains_hitbox()``,
    que respeta el ``overlap_threshold`` configurado por región:
    - Si ``overlap_threshold == 0.0``: usa el punto central de la hitbox
      (comportamiento original).
    - Si ``overlap_threshold > 0.0``: evalúa la fracción del área de la
      hitbox que interseca con la región.

    Attributes:
        regions: Lista de regiones de interés.
        hitbox_size: Semilado de la hitbox cuadrada en píxeles.
        states: Diccionario ``{region_id: ZoneState}`` con el estado de
            cada región.
    """

    def __init__(self, region_manager, hitbox_size: int = 0):
        """Inicializa la lógica de eventos.

        Args:
            region_manager: Objeto ``RegionManager`` con la lista de regiones.
            hitbox_size: Semilado de la hitbox en píxeles, usado para calcular
                el solapamiento de área cuando ``overlap_threshold > 0.0``.
                Si es 0, el solapamiento por área no tiene efecto aunque el
                umbral de la región sea mayor que cero.
        """
        self.regions = region_manager.regions
        self.hitbox_size = hitbox_size
        self.states = {r.region_id: ZoneState() for r in self.regions}

    def update(self, position, t: float, frame_idx: int) -> None:
        """Actualiza el estado de todas las regiones dado un nuevo frame.

        Evalúa si el objeto entra o sale de cada región y actualiza
        contadores y tiempos acumulados. La decisión de pertenencia usa
        ``contains_hitbox``, que delega en el modo punto o área según
        el ``overlap_threshold`` de cada región.

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
            inside_now = region.contains_hitbox(position, self.hitbox_size)

            # Evento de entrada
            if inside_now and not state.inside:
                state.inside = True
                state.enter_time = t
                state.enter_frame.append(frame_idx)
                state.entries += 1

            # Evento de salida
            elif not inside_now and state.inside:
                state.inside = False
                state.total_time += t - state.enter_time
                state.exit_frame.append(frame_idx)
                state.events.append((state.enter_time, t))
                state.enter_time = None