# DungeonMice
![alt text](img/logo.jpeg)

Herramienta de tracking automático de ratones en laberintos para experimentos conductuales. Detecta la posición del ratón frame a frame, registra entradas y salidas de regiones de interés, y genera reportes en Excel con métricas por región (latencia, tiempo, distancia, número de entradas) junto con imágenes de trayectoria y mapas de calor.

## Laberintos soportados

- **CrossMaze** — laberinto en cruz con brazos abiertos y cerrados
- **MorrisPool** — piscina de Morris con cuadrante objetivo

## Instalación

```bash
pip install -r requirements.txt
```

> Requiere `opencv-contrib-python` (no `opencv-python`) para el tracker CSRT.

## Uso

1. Configurar el experimento en `src/input.py` — definir el tipo de laberinto, las regiones de interés con sus coordenadas, y los parámetros de video.
2. Desde `src/`, correr:

```python
from input import input1
from RunExperiment import main

main(input1)
```

3. Los resultados (Excel, imágenes de trayectoria, heatmap) se guardan en la carpeta del video.

## Configuración de regiones

Las regiones se definen en `src/input.py` con coordenadas en píxeles relativas al frame del video. Cada región tiene un `overlap_threshold` que controla qué fracción de la hitbox debe estar dentro para registrar una entrada.

```python
PolygonRegion("este", [[630,420],[900,420],[900,350],[630,350]], overlap_threshold=0.80)
```

Para asignar un `start_time` distinto a cada video, crear un `config.txt` en la carpeta de videos:

```
nombre_video = 25.0
otro_video   = 16.0
```

## Parámetros principales

| Parámetro | Descripción |
|-----------|-------------|
| `min_detection_area` | Área mínima del blob (px²). 400 detecta ratones delgados. |
| `hitbox_size` | Semilado de la hitbox cuadrada en píxeles. |
| `start_time` | Segundos de warmup antes de empezar a registrar. |
| `mog_threshold` | Sensibilidad de MOG2. Subir si hay muchos falsos positivos. |
| `blur_size` | GaussianBlur antes de MOG2. 0 = sin blur. |
| `use_csrt` | Tracker CSRT como primario (trayectoria más suave). Default True. |

## Estructura

```
DungeonMice/
├── src/
│   ├── RunExperiment.py       — orquestador principal
│   ├── tracker.py             — detección MOG2 + CSRT
│   ├── logic.py               — lógica de entradas/salidas por región
│   ├── regions.py             — definición de regiones (polígono, círculo, fracción)
│   ├── visualizer.py          — visualización y guardado de imágenes
│   ├── input.py               — configuración del experimento
│   └── labyrinths/
│       ├── labyrinth.py           — clase base
│       ├── labyrinth_CrossMaze.py
│       └── labyrinth_MorrisPool.py
└── videos/                    — carpeta de videos (no incluida en el repo)
```