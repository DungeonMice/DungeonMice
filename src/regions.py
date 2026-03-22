import numpy as np
import cv2
import math

"""
regions.py
==========

Definición de regiones de interés (ROI) geométricas para procesamiento
de imágenes y visión por computador usando OpenCV.

Este módulo define una interfaz común para regiones y varias implementaciones:
- PolygonRegion: regiones definidas por polígonos (incluye rectángulos).
- CircleRegion: regiones definidas por círculos.
- CircularFractionRegion: sectores de círculo.

Cada región implementa:
- contains(point): prueba de pertenencia
- mask(shape): generación de máscara binaria
- draw(frame): dibujo sobre un frame
"""


class Region:
    """
    Clase base abstracta para una región de interés (ROI).

    Todas las regiones geométricas deben implementar esta interfaz.
    La clase NO maneja tiempo, eventos ni lógica externa.
    """

    def contains(self, point: tuple[float, float]) -> bool:
        """
        Determina si un punto pertenece a la región.

        Args:
            point (tuple[float, float]): Punto en coordenadas de imagen.

        Returns:
            bool: True si el punto está dentro de la región.
        """
        raise NotImplementedError

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Genera una máscara binaria de la región.

        Args:
            shape (tuple[int, int]): Dimensiones (alto, ancho) de la imagen.

        Returns:
            np.ndarray: Imagen uint8 con valores:
                - 255 dentro de la región
                - 0 fuera de la región
        """
        raise NotImplementedError

    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> None:
        """
        Dibuja la región sobre un frame.

        Args:
            frame (np.ndarray): Imagen sobre la cual dibujar.
            color (tuple[int, int, int]): Color del contorno en formato (B, G, R).
            thickness (int): Grosor del contorno.
        """
        raise NotImplementedError


class PolygonRegion(Region):
    """
    Región de interés (ROI) definida por un polígono.

    Esta clase representa cualquier región poligonal:
    - rectángulos
    - polígonos convexos
    - polígonos cóncavos

    Los puntos se almacenan en formato compatible con OpenCV: (N, 1, 2).
    """

    def __init__(self, region_id, points):
        """
        Inicializa una región poligonal.

        Args:
            region_id (str | int): Identificador único de la región.
            points (iterable[tuple[int, int]]): Vértices del polígono en orden.

        Raises:
            ValueError: Si hay menos de 3 puntos.
        """
        self.region_id = region_id

        self.points = (
            np.array(points, dtype=np.int32)
            .reshape((-1, 1, 2))
        )

        if self.points.shape[0] < 3:
            raise ValueError("Un polígono debe tener al menos 3 puntos")

    def contains(self, point):
        """
        Determina si un punto está dentro o sobre el borde del polígono.

        Args:
            point (tuple[float, float]): Punto en coordenadas de imagen.

        Returns:
            bool: True si está dentro o en el borde.
        """
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.points, pt, False) >= 0

    def mask(self, shape):
        """
        Genera una máscara binaria del polígono.

        Args:
            shape (tuple[int, int]): Dimensiones de la imagen.

        Returns:
            np.ndarray: Máscara binaria.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.points], -1, 255, -1)
        return mask

    def draw(self, frame, color=(0, 255, 0), thickness=2):
        """
        Dibuja el contorno del polígono sobre un frame.

        Args:
            frame (np.ndarray): Imagen destino.
            color (tuple[int, int, int]): Color del contorno.
            thickness (int): Grosor de línea.
        """
        cv2.polylines(frame, [self.points], True, color, thickness)


class CircleRegion(Region):
    """
    Región de interés (ROI) definida por un círculo.
    """

    def __init__(self, region_id, center, radius):
        """
        Inicializa una región circular.

        Args:
            region_id (str | int): Identificador único.
            center (tuple[float, float]): Centro del círculo.
            radius (float): Radio en píxeles.

        Raises:
            ValueError: Si el radio no es positivo.
        """
        self.region_id = region_id
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)

        if self.radius <= 0:
            raise ValueError("El radio debe ser positivo")

    def contains(self, point):
        """
        Determina si un punto está dentro del círculo.

        Args:
            point (tuple[float, float]): Punto a evaluar.

        Returns:
            bool: True si está dentro o en el borde.
        """
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius

    def mask(self, shape):
        """
        Genera una máscara del círculo.

        Args:
            shape (tuple[int, int]): Dimensiones de la imagen.

        Returns:
            np.ndarray: Máscara binaria.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        center_int = tuple(map(int, self.center))
        cv2.circle(mask, center_int, int(self.radius), 255, -1)
        return mask

    def draw(self, frame, color=(0, 255, 0), thickness=2):
        """
        Dibuja el círculo.

        Args:
            frame (np.ndarray): Imagen destino.
            color (tuple[int, int, int]): Color del contorno.
            thickness (int): Grosor.
        """
        center_int = tuple(map(int, self.center))
        cv2.circle(frame, center_int, int(self.radius), color, thickness)


class CircularFractionRegion(Region):
    """
    Región definida por una fracción angular de un círculo.
    """

    def __init__(self, region_id, center, radius, angle_start=0.0, angle_end=None, fraction=None):
        """
        Inicializa una región tipo sector circular.

        Args:
            region_id (str | int): Identificador único.
            center (tuple[float, float]): Centro del círculo.
            radius (float): Radio.
            angle_start (float): Ángulo inicial en grados.
            angle_end (float | None): Ángulo final.
            fraction (float | None): Fracción del círculo (0, 1].

        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        self.region_id = region_id
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)

        if self.radius <= 0:
            raise ValueError("El radio debe ser positivo")

        angle_start = float(angle_start) % 360

        if fraction is not None:
            if not (0 < fraction <= 1):
                raise ValueError("fraction debe estar en (0, 1]")
            angle_end = angle_start + 360.0 * float(fraction)

        if angle_end is None:
            raise ValueError("Debes definir angle_end o fraction")

        self.angle_start = angle_start % 360
        self.angle_end = float(angle_end)
    
    def _angle_in_sector(self, angle):
        if self.angle_start <= self.angle_end:
            return self.angle_start <= angle <= self.angle_end
        else:
            return angle >= self.angle_start or angle <= self.angle_end

    def contains(self, point):
        """
        Verifica si un punto está dentro del sector.

        Args:
            point (tuple[float, float]): Punto a evaluar.

        Returns:
            bool: True si pertenece al sector.
        """
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]

        if dx * dx + dy * dy > self.radius * self.radius:
            return False

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return self._angle_in_sector(angle)

    def mask(self, shape):
        """
        Genera máscara del sector circular.

        Args:
            shape (tuple[int, int]): Dimensiones.

        Returns:
            np.ndarray: Máscara binaria.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        center_int = tuple(map(int, self.center))

        cv2.ellipse(
            mask,
            center_int,
            (int(self.radius), int(self.radius)),
            0,
            self.angle_start,
            self.angle_end,
            255,
            -1,
        )
        return mask

    def draw(self, frame, color=(0, 255, 0), thickness=2):
        """
        Dibuja el sector.

        Args:
            frame (np.ndarray): Imagen destino.
            color (tuple[int, int, int]): Color.
            thickness (int): Grosor.
        """
        center_x, center_y = self.center
        cx, cy = int(round(center_x)), int(round(center_y))

        steps = 100
        a_start = math.radians(self.angle_start)
        a_end = math.radians(self.angle_end)

        arc_points = []
        for i in range(steps + 1):
            t = a_start + (a_end - a_start) * i / steps
            x = int(round(center_x + self.radius * math.cos(t)))
            y = int(round(center_y + self.radius * math.sin(t)))
            arc_points.append((x, y))

        points = np.array([(cx, cy)] + arc_points + [(cx, cy)], dtype=np.int32)
        pts = points.reshape((-1, 1, 2))

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


class RegionManager:
    """
    Contenedor de múltiples regiones de interés.
    """

    def __init__(self, regions):
        """
        Inicializa el gestor de regiones.

        Args:
            regions (list[Region]): Lista de regiones de interés.
        """
        self.regions = regions