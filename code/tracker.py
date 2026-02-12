import cv2
import numpy as np

class MouseTracker:
    """
    Detector de posición del ratón basado en sustracción de fondo.
    Esta clase se encarga exclusivamente de localizar la posición del
    objeto (ratón) en un frame en escala de grises usando técnicas
    clásicas de visión por computadora.
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
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = min_area
        self.trajectory = []
        self.recording = False
    
    def start_recording(self):
        """
        Activa la grabación de la trayectoria.
        
        A partir de este momento, cada posición detectada se guardará
        en self.trajectory.
        """
        self.recording = True
        
    def locate(self, gray_frame):
        """
        Localiza la posición del ratón en un frame.

        Aplica sustracción de fondo, filtrado morfológico y detección
        de contornos para estimar la posición del objeto.

        Parámetros
        ----------
        gray_frame : np.ndarray
            Frame en escala de grises.

        Retorna
        -------
        center_real : tuple or None
            Coordenadas (x, y) del centro del ratón según el contorno
            detectado. Útil para dibujar la hitbox en tiempo real.

        fgmask : np.ndarray
            Máscara binaria resultante de la sustracción de fondo,
            útil para depuración o visualización.
        """
        fgmask = self.bg.apply(gray_frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel) # rellena huecos dentro del ratón
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)  # elimina ruido pequeño
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return None, fgmask
    
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area:
            return None, fgmask
        
        # Centroide del contorno
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, fgmask
        # Ahora solo guardamos la posición si estamos en modo grabación, para evitar guardar posiciones erráticas antes de empezar a dibujar.
        if self.recording: 
            cx_real = int(M["m10"]/M["m00"])
            cy_real = int(M["m01"]/M["m00"])
            center_real = (cx_real, cy_real)
            self.trajectory.append(center_real)
            return center_real, fgmask
        return None, fgmask
    
    def get_total_distance(self, start_frame=0):
        """
        Calcula la distancia total recorrida en píxeles.
        
        Parámetros
        ----------
        start_frame : int
            Frame desde el cual empezar a contar (default: 0).
            Útil para excluir movimientos iniciales.
        
        Retorna
        -------
        float
            Distancia total en píxeles.
        """
        if len(self.trajectory) <= start_frame + 1:
            return 0.0
        
        trajectory_filtered = self.trajectory[start_frame:]
        total_distance = 0.0
        
        for i in range(1, len(trajectory_filtered)):
            pt1 = trajectory_filtered[i-1]
            pt2 = trajectory_filtered[i]
            
            # Distancia euclidiana entre dos puntos sqrt((x2-x1)² + (y2-y1)²)
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            total_distance += distance
        
        return total_distance
    
    
