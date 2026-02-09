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
        self.prev_pos = None
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
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel) #prueba
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
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
        cx_real = int(M["m10"]/M["m00"])
        cy_real = int(M["m01"]/M["m00"])
        center_real = (cx_real, cy_real)
        # Solo guardar si estamos grabando
        if self.recording:
            self.trajectory.append(center_real) 
        self.prev_pos = center_real
        return center_real, fgmask

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
    
    def save_trajectory_image(self, video_cap, regions, output_filename=None):
        """
        Guarda una imagen con la trayectoria completa sobre el primer frame.
        
        Parámetros
        ----------
        video_cap : cv2.VideoCapture
            Objeto de captura de video para obtener el primer frame.
        regions : RegionManager
            Regiones a dibujar en la imagen.
        output_filename : str, optional
            Nombre del archivo de salida. Si es None, se genera automáticamente.
        
        Retorna
        -------
        str or None
            Nombre del archivo guardado, o None si falló.
        """
        # Obtener primer frame
        current_pos = video_cap.get(cv2.CAP_PROP_POS_FRAMES)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, background = video_cap.read()
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # Restaurar posición
        
        if not ret or len(self.trajectory) < 2:
            return None
        
        # Dibujar regiones
        for region in regions.regions:
            region.draw(background, (0, 255, 0), 2)
        
        # Dibujar trayectoria completa
        for i in range(1, len(self.trajectory)):
            pt1 = self.trajectory[i-1]
            pt2 = self.trajectory[i]
            cv2.line(background, pt1, pt2, (255, 0, 255), 2)
        
        # Marcar inicio (verde) y fin (rojo)
        cv2.circle(background, self.trajectory[0], 8, (0, 255, 0), -1)
        cv2.circle(background, self.trajectory[-1], 8, (0, 0, 255), -1)
        
        # Agregar texto con la distancia total
        total_distance = self.get_total_distance()
        text = f"Distancia: {total_distance:.0f} px"
        cv2.putText(background, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Guardar imagen
        if output_filename is None:
            output_filename = f"trajectory_{id(self)}.png"
        
        cv2.imwrite(output_filename, background)
        print(f"Guardado: {output_filename}")
        
        return output_filename 
