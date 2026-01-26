class ZoneState:
    """
    Estado temporal asociado a una región de interés.

    Esta clase no conoce geometría ni video. Únicamente almacena
    el historial lógico de una región: si el objeto está dentro,
    cuándo entró, cuántas veces entró y cuánto tiempo total permaneció.

    Atributos
    ----------
    inside : bool
        Indica si el objeto se encuentra actualmente dentro de la región.
    enter_time : float or None
        Timestamp (en segundos) del último ingreso a la región.
        Es None si el objeto no está dentro.
    total_time : float
        Tiempo total acumulado (en segundos) que el objeto ha pasado
        dentro de la región.
    entries : int
        Número total de veces que el objeto ha entrado a la región.
    """
    def __init__(self):
        self.inside = False
        self.enter_time = None
        self.total_time = 0
        self.entries = 0


class EventLogic:
    """
    Lógica de eventos de entrada y salida para múltiples regiones.

    Esta clase recibe posiciones del objeto a lo largo del tiempo
    y determina eventos de entrada y salida en cada región de interés.
    No realiza detección visual ni define geometría: solo coordina estados.

    Supuestos
    ---------
    - `region_manager.regions` es una lista de regiones con:
        - atributo `id` único
        - método `contains(position)` -> bool
    - El tiempo `t` está dado en segundos y es monótono creciente.

    Atributos
    ----------
    regions : list
        Lista de regiones de interés.
    states : dict
        Diccionario que mapea region.id -> ZoneState.
    """
    def __init__(self, region_manager):
        self.regions = region_manager.regions
        self.states = {r.region_id: ZoneState() for r in self.regions}

    def update(self, position, t):
        """
        Actualiza el estado de todas las regiones dado un nuevo frame.

        Evalúa si el objeto entra o sale de cada región y actualiza
        contadores y tiempos acumulados según corresponda.

        Parámetros
        ----------
        position : tuple or None
            Posición (x, y) del objeto detectado en el frame actual.
            Si es None, no se actualiza ningún estado.
        t : float
            Timestamp actual en segundos (por ejemplo, frame / fps).

        Retorna
        -------
        None
        """
        if position is None:
            return

        for region in self.regions:
            state = self.states[region.region_id]
            inside_now = region.contains(position)

            # Evento de entrada
            if inside_now and not state.inside:
                state.inside = True
                state.enter_time = t
                state.entries += 1

            # Evento de salida
            elif not inside_now and state.inside:
                state.inside = False
                state.total_time += t - state.enter_time
                state.enter_time = None
