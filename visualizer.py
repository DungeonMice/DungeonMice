import cv2
import numpy as np
"""
visualizer.py

Módulo de visualización para el de tracking de ratones.

Este archivo se encarga de:
- Dibujar regiones de interés con colores según su estado
- Dibujar hitbox del ratón
- Dibujar trayectoria sobre la máscara de detección
- Mostrar ventanas de visualización
"""
class ExperimentVisualizer:
    """
    Clase para manejar toda la visualización del experimento.
    
    Atributos
    ----------
    regions : RegionManager
        Gestor de regiones de interés a visualizar.
    hitbox_size : int
        Tamaño de la hitbox del ratón en píxeles.
    """
    
    def __init__(self, regions, hitbox_size=10):
        """
        Inicializa el visualizador.
        
        Parámetros
        ----------
        regions : RegionManager
            Gestor de regiones de interés.
        hitbox_size : int
            Tamaño de la hitbox del ratón (default: 10).
        """
        self.regions = regions
        self.hitbox_size = hitbox_size
    
    def draw_regions(self, frame, logic_states):
        """
        Dibuja todas las regiones sobre el frame con color según su estado.
        
        Parámetros
        ----------
        frame : np.ndarray
            Frame sobre el cual dibujar.
        logic_states : dict
            Diccionario {region_id: ZoneState} con el estado de cada región.
        """
        for region in self.regions.regions:
            state = logic_states[region.region_id]
            
            if state.inside:
                color = (0, 0, 255)   # rojo si el ratón está dentro
            else:
                color = (0, 255, 0)   # verde si está fuera
            
            region.draw(frame, color)
    
    def draw_hitbox(self, frame, position, logic_states):
        """
        Dibuja la hitbox (rectángulo) alrededor de la posición del ratón.
        
        El color depende de si el ratón está dentro de alguna región:
        - Rojo si está dentro de al menos una región
        - Verde si no está en ninguna región
        
        Parámetros
        ----------
        frame : np.ndarray
            Frame sobre el cual dibujar.
        position : tuple or None
            Coordenadas (x, y) del ratón. Si es None, no dibuja nada.
        logic_states : dict
            Diccionario {region_id: ZoneState} con el estado de cada región.
        """
        if position is None:
            return
        
        inside_any = any(logic_states[r.region_id].inside for r in self.regions.regions)
        hitbox_color = (0, 0, 255) if inside_any else (0, 255, 0)
        
        x, y = position
        cv2.rectangle(
            frame,
            (x - self.hitbox_size, y - self.hitbox_size),
            (x + self.hitbox_size, y + self.hitbox_size),
            hitbox_color,
            2
        )
    
    def draw_trajectory_on_mask(self, fgmask, trajectory, frame_idx, draw_start_frame):
        """
        Dibuja la trayectoria acumulada sobre la máscara de detección.
        
        Convierte la máscara a color y dibuja líneas conectando los puntos
        de la trayectoria.
        
        Parámetros
        ----------
        fgmask : np.ndarray
            Máscara binaria de detección (escala de grises).
        trajectory : list
            Lista de puntos (x, y) de la trayectoria.
        frame_idx : int
            Índice del frame actual.
        draw_start_frame : int
            Frame a partir del cual se debe empezar a dibujar.
        
        Retorna
        -------
        np.ndarray
            Máscara con la trayectoria dibujada (en color si hay trayectoria,
            en escala de grises o color según el estado).
        """
        # Dibujar trayectoria si ya hay puntos grabados
        if len(trajectory) > 1:
            fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i-1]
                pt2 = trajectory[i]
                cv2.line(fgmask_color, pt1, pt2, (255, 255, 0), 2)  # cyan
            
            return fgmask_color
        else:
            # Convertir máscara a color después del delay para mantener consistencia
            if frame_idx >= draw_start_frame:
                return cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            else:
                return fgmask