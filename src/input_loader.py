"""input_loader.py
=================
Este script está encargado de leer, cargar y validar la configuración de un experimento desde un archivo JSON.
El json se lee y se si todo es correcto, devuelve un objeto tipo ``Labyrinth`` para su uso en ``RunExperiment.main()``.
"""

import json
from pathlib import Path

from regions import (
    RegionManager,
    PolygonRegion,
    CircleRegion,
    CircularFractionRegion,
)
from labyrinths.labyrinth_CrossMaze import CrossMaze
from labyrinths.labyrinth_MorrisPool import MorrisPool
from labyrinths.labyrinth_BarnesMaze import BarnesMaze


# ---------------------------------------------------------------------------
# Registro de laberintos disponibles
# ---------------------------------------------------------------------------

_LABYRINTH_REGISTRY: dict[str, type] = {
    "CrossMaze":  CrossMaze,
    "MorrisPool": MorrisPool,
    "BarnesMaze": BarnesMaze,
}

# ---------------------------------------------------------------------------
# Defaults para parámetros opcionales del tracker
# ---------------------------------------------------------------------------

_TRACKER_DEFAULTS: dict[str, object] = {
    "kernel_size":        5,
    "blur_size":          0,
    "mog_threshold":      30,
    "recording_lr":       0.002,
    "use_csrt":           False,
    "mog_history":        500,
    "end_time":           None,
}


# ---------------------------------------------------------------------------
# Constructores de regiones
# ---------------------------------------------------------------------------

def _build_region(raw: dict):
    """Construye una instancia de ``Region`` a partir de un diccionario.

    Args:
        raw: Diccionario con al menos las claves ``"type"`` e ``"id"``.
            Las claves restantes dependen del tipo de región.

    Returns:
        Una instancia de ``PolygonRegion``, ``CircleRegion`` o
        ``CircularFractionRegion``.

    Raises:
        ValueError: Si ``"type"`` no corresponde a ninguna región conocida,
            o si faltan claves obligatorias para el tipo indicado.
    """
    region_type = raw.get("type")
    region_id   = raw.get("id")
    threshold   = float(raw.get("overlap_threshold", 0.0))

    if region_type == "PolygonRegion":
        _require_keys(raw, ["points"], context=f"región '{region_id}'")
        return PolygonRegion(region_id, raw["points"], overlap_threshold=threshold)

    if region_type == "CircleRegion":
        _require_keys(raw, ["center", "radius"], context=f"región '{region_id}'")
        return CircleRegion(region_id, tuple(raw["center"]), float(raw["radius"]), overlap_threshold=threshold,)

    if region_type == "CircularFractionRegion":
        _require_keys(raw, ["center", "radius"], context=f"región '{region_id}'")
        return CircularFractionRegion(region_id, tuple(raw["center"]), float(raw["radius"]), angle_start=float(raw.get("angle_start", 0.0)),
            angle_end=raw.get("angle_end"),          # None si no se especifica
            fraction=raw.get("fraction"),             # None si no se especifica
            overlap_threshold=threshold,
        )

    raise ValueError(
        f"Tipo de región desconocido: '{region_type}'. "
        f"Tipos válidos: PolygonRegion, CircleRegion, CircularFractionRegion."
    )


# ---------------------------------------------------------------------------
# Constructor principal
# ---------------------------------------------------------------------------

def load_experiment(json_path: str | Path):
    """Carga un archivo ``.json`` y devuelve el objeto ``Labyrinth`` listo para usar.

    El JSON debe tener la siguiente estructura (las claves con * son opcionales
    y usan los valores por defecto indicados)::

        {
            "labyrinth_type": "CrossMaze",       // CrossMaze | MorrisPool | BarnesMaze
            "video_path":     "../videos/Cross",
            "treatment":      "Control",
            "subject_id":     "1",
            "start_time":     25,
            "regions": [
                {
                    "type":              "PolygonRegion",
                    "id":                "norte",
                    "points":            [[545,350],[635,350],[635,10],[545,10]],
                    "overlap_threshold": 0.80       // * default 0.0
                }
            ],
            "min_detection_area": 800,
            "hitbox_size":        40,
            "end_time":           null,            // * default null
            "kernel_size":        15,              // * default 5
            "blur_size":          9,               // * default 0
            "mog_threshold":      35,              // * default 30
            "recording_lr":       0.003,           // * default 0.002
            "use_csrt":           false,           // * default false
            "mog_history":        500              // * default 500
        }

    Args:
        json_path: Ruta al archivo ``.json`` de configuración del experimento.

    Returns:
        Una instancia del ``Labyrinth`` correspondiente al ``labyrinth_type``
        indicado en el JSON.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan claves obligatorias o los valores son inválidos.
        json.JSONDecodeError: Si el archivo no es JSON válido.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    # --- Claves obligatorias ---
    _require_keys(cfg, [
        "labyrinth_type", "video_path", "treatment", "subject_id",
        "start_time", "regions", "min_detection_area", "hitbox_size",
    ])

    labyrinth_type = cfg["labyrinth_type"]
    if labyrinth_type not in _LABYRINTH_REGISTRY:
        raise ValueError(
            f"labyrinth_type desconocido: '{labyrinth_type}'. "
            f"Tipos válidos: {list(_LABYRINTH_REGISTRY)}"
        )

    # --- Construir regiones ---
    if not cfg["regions"]:
        raise ValueError("El campo 'regions' no puede estar vacío.")

    regions = RegionManager([_build_region(r) for r in cfg["regions"]])

    # --- Resolver parámetros opcionales con sus defaults ---
    opts = {key: cfg.get(key, default) for key, default in _TRACKER_DEFAULTS.items()} # Si la clave existe en cfg, se usa su valor; si no, se usa el default.

    # --- Instanciar el Labyrinth ---
    LabyrinthClass = _LABYRINTH_REGISTRY[labyrinth_type]
    return LabyrinthClass(
        video_path=cfg["video_path"],
        treatment=cfg["treatment"],
        subject_id=cfg["subject_id"],
        regions=regions,
        min_detection_area=int(cfg["min_detection_area"]),
        hitbox_size=int(cfg["hitbox_size"]),
        start_time=float(cfg["start_time"]),
        **opts,
    )


# ---------------------------------------------------------------------------
# Utilidad interna
# ---------------------------------------------------------------------------

def _require_keys(d: dict, keys: list[str], context: str = "JSON raíz") -> None:
    """Verifica que todas las claves estén presentes en el diccionario.

    Args:
        d: Diccionario a verificar.
        keys: Lista de claves que deben existir.
        context: Descripción del contexto para mensajes de error.

    Raises:
        ValueError: Si alguna clave falta, listando todas las que faltan.
    """
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Faltan claves obligatorias en {context}: {missing}")
