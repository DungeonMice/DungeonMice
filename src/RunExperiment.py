import os
import cv2
import numpy as np
from tracker import MouseTracker
from logic import EventLogic
from visualizer import ExperimentVisualizer


def load_config(folder):
    """
    Lee el archivo config.txt de la carpeta si existe.

    Formato esperado (una línea por video, sin extensión):
        nombre_video = start_time

    Parámetros
    ----------
    folder : str
        Ruta a la carpeta donde buscar el config.txt.

    Retorna
    -------
    dict
        {nombre_video: start_time} o {} si no existe el archivo.
    """
    config = {}
    config_path = os.path.join(folder, "config.txt")
    if not os.path.exists(config_path):
        print("No se encontró config.txt — usando start_time del input para todos los videos.")
        return config
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, value = line.split("=", 1)
            config[name.strip()] = float(value.strip())
    print(f"Config cargado: {len(config)} videos con start_time personalizado.")
    return config


def process_single_video(video_path, labyrinth):
    """
    Procesa un único video y retorna los eventos registrados.

    Parámetros
    ----------
    video_path : str
        Ruta al archivo de video.
    labyrinth : Labyrinth
        Objeto del experimento con sus parámetros.

    Retorna
    -------
    tuple (events, trajectory_x, trajectory_y, first_frame)
        events        : dict con los estados de cada región
        trajectory_x  : lista de coordenadas x de la trayectoria
        trajectory_y  : lista de coordenadas y de la trayectoria
        first_frame   : primer frame del video para generar imagen de trayectoria
    """
    # Reiniciar trayectoria para este video
    labyrinth.trajectory_x    = []
    labyrinth.trajectory_y    = []
    labyrinth.trajectory_time = []

    # --- Abrir video ---
    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labyrinth.fps = fps

    # Guardar primer frame antes del loop para imagen de trayectoria
    ret, first_frame = cap.read()
    if not ret:
        print(f"No se pudo leer el primer frame de {video_path}")
        return None, [], [], None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rebobinar al inicio

    frame_idx  = 0
    tracker    = MouseTracker(
        min_area=labyrinth.min_detection_area,
        kernel_size=labyrinth.kernel_size,
        blur_size=labyrinth.blur_size
    )
    logic      = EventLogic(labyrinth.regions)
    visualizer = ExperimentVisualizer(labyrinth.regions, hitbox_size=labyrinth.hitbox_size)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", width, height)
    cv2.namedWindow("fgmask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("fgmask", width, height)

    # --- Loop principal ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        t = frame_idx / fps
        
        # Conversión a escala de grises para el detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_frame = int(labyrinth.start_time * fps)
        # Activar grabación de trayectoria cuando se alcance el frame indicado
        if frame_idx == start_frame:
            tracker.start_recording()

        # Localizar la posición del ratón
        pos_real, fgmask = tracker.locate(gray)
        trajectory = tracker.trajectory

        # Actualizar la lógica de eventos solo después del start_time
        if frame_idx >= start_frame:
            logic.update(pos_real, t, frame_idx)

        # Visualización
        fgmask = visualizer.draw_trajectory_on_mask(fgmask, tracker.trajectory, frame_idx, labyrinth.start_time * fps)
        visualizer.draw_regions(frame, logic.states)
        visualizer.draw_hitbox(frame, pos_real, logic.states)

        # Procesamiento del frame
        labyrinth.process_frame(position=trajectory, time=t)

        # Mostrar nombre del video y timestamp en el frame
        cv2.putText(frame, os.path.basename(video_path), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"t = {t:.2f} s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("frame", frame)
        cv2.imshow("fgmask", fgmask)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

    return (
        logic.states,
        list(labyrinth.trajectory_x),
        list(labyrinth.trajectory_y),
        first_frame,
    )


def main(labyrinth):
    """
    Procesa uno o varios videos según video_path sea archivo o carpeta.
    Si existe un config.txt en la carpeta, usa el start_time definido
    por video. Si no, usa el start_time del input como default para todos.
    Al finalizar llama process_video() con los resultados de todos los videos.

    Parámetros
    ----------
    labyrinth : Labyrinth
        Objeto del experimento con sus parámetros.
    """
    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_path = labyrinth.video_path

    # --- Determinar lista de videos a procesar ---
    if os.path.isdir(video_path):
        folder = video_path
        videos = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ])
        if not videos:
            print(f"No se encontraron videos en {folder}")
            return
    else:
        folder = os.path.dirname(video_path)
        videos = [video_path]

    # --- Cargar config de start_times si existe ---
    config = load_config(folder)
    default_start_time = labyrinth.start_time  # guardar default antes del loop

    # --- Procesar cada video ---
    all_results      = {}
    all_trajectories = {}
    all_video_paths  = {}
    all_first_frames = {}
    all_start_times  = {}

    for video_file in videos:
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # Usar start_time del config si existe, sino el default del input
        labyrinth.start_time = config.get(video_name, default_start_time)
        print(f"\nProcesando: {video_name} (start_time={labyrinth.start_time}s)")

        events, traj_x, traj_y, first_frame = process_single_video(video_file, labyrinth)

        all_results[video_name]      = events
        all_trajectories[video_name] = (traj_x, traj_y)
        all_video_paths[video_name]  = video_file
        all_first_frames[video_name] = first_frame
        all_start_times[video_name] = labyrinth.start_time  # guardar antes de que cambie
    # --- Finalización ---
    labyrinth.process_video(
        all_results=all_results,
        all_trajectories=all_trajectories,
        all_video_paths=all_video_paths,
        all_first_frames=all_first_frames,
        all_start_times=all_start_times
    )