import numpy as np
import cv2


class Region:
    """
    Representa una región de interés (ROI) definida por un polígono.

    Esta clase encapsula únicamente la geometría de la región y las
    operaciones básicas asociadas a ella. No conoce lógica temporal,
    eventos ni detección de objetos.

    Atributos
    ----------
    id : str or int
        Identificador único de la región.
    points : np.ndarray
        Array de puntos (Nx2) que definen el contorno del polígono
        en coordenadas de imagen (x, y).
    """
    def __init__(self, region_id, points):
        """
        Parámetros
        ----------
        region_id : str or int
            Identificador único de la región.
        points : iterable
            Lista o array de puntos [(x1, y1), (x2, y2), ...]
            que definen el contorno de la región.
        """
        self.id = region_id
        self.points = np.array(points, dtype=np.int32)

    def contains(self, point):
        """
        Determina si un punto se encuentra dentro de la región.

        Parámetros
        ----------
        point : tuple
            Punto (x, y) en coordenadas de imagen.

        Retorna
        -------
        bool
            True si el punto está dentro o sobre el borde del polígono,
            False si está fuera.
        """
        return cv2.pointPolygonTest(self.points, point, False) >= 0

    def mask(self, shape):
        """
        Genera una máscara binaria de la región.

        La máscara tiene valor 255 dentro de la región y 0 fuera de ella.
        Se usa típicamente para filtrar regiones en una imagen.

        Parámetros
        ----------
        shape : tuple
            Forma de la máscara (alto, ancho), usualmente frame.shape[:2].

        Retorna
        -------
        np.ndarray
            Imagen binaria (uint8) con la región dibujada.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.points], -1, 255, -1)
        return mask


class RegionManager:
    """
    Contenedor de múltiples regiones de interés.

    Esta clase existe para centralizar el manejo de regiones y
    proporcionar una interfaz común al resto del backend.
    No implementa lógica ni eventos.
    """
    def __init__(self, regions):
        """
        Parámetros
        ----------
        regions : list of Region
            Lista de regiones de interés.
        """
        self.regions = regions
