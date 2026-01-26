import cv2
import numpy as np


class MouseTracker:
    """
    Detector de posición del ratón basado en sustracción de fondo.

    Esta clase se encarga exclusivamente de localizar la posición del
    objeto (ratón) en un frame en escala de grises usando técnicas
    clásicas de visión por computadora.

    No conoce regiones de interés, tiempos ni lógica de eventos.
    Su única salida es la posición estimada del objeto y la máscara
    binaria asociada a la detección.
    """
    def __init__(self, min_area=4000):
        """
        Inicializa el detector.

        Parámetros
        ----------
        min_area : int
            Área mínima (en píxeles) que debe tener un contorno para
            ser considerado como el ratón. Sirve para filtrar ruido.
        """
        self.bg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        self.min_area = min_area

    def locate(self, gray_frame):
        """
        Localiza la posición del ratón en un frame.

        Aplica sustracción de fondo, filtrado morfológico y detección
        de contornos para estimar la posición del objeto como el centro
        del contorno de mayor área.

        Parámetros
        ----------
        gray_frame : np.ndarray
            Frame en escala de grises.

        Retorna
        -------
        center : tuple or None
            Coordenadas (x, y) del centro del ratón si fue detectado.
            Retorna None si no se detecta ningún objeto válido.
        fgmask : np.ndarray
            Máscara binaria resultante de la sustracción de fondo,
            útil para depuración o visualización.
        """
        fgmask = self.bg.apply(gray_frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        cnts = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        if not cnts:
            return None, fgmask

        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area:
            return None, fgmask

        #x, y, w, h = cv2.boundingRect(cnt)
        #center = (x + w // 2, y + h // 2)
        
        #Mejor centroide del contorno no el bounding box
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, fgmask

        center = (
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        )

        return center, fgmask
