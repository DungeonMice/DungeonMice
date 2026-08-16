from labyrinths.labyrinth_MorrisPool import MorrisPool
from labyrinths.labyrinth_CrossMaze import CrossMaze
from labyrinths.labyrinth_BarnesMaze import BarnesMaze
from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion

input1 = CrossMaze(
    video_path="../videos/CrossMaze",
    treatment="Control",
    subject_id="1",
    regions=RegionManager([
        PolygonRegion("este",  [[630,420],[900,420],[900,350],[630,350]], overlap_threshold=0.80),
        PolygonRegion("oeste", [[240,420],[240,350],[550,350],[550,420]], overlap_threshold=0.80),
        PolygonRegion("norte", [[545,350],[635,350],[635,10],[545,10]],   overlap_threshold=0.80),
        PolygonRegion("sur",   [[545,720],[635,720],[635,420],[545,420]], overlap_threshold=0.80)
    ]),
    min_detection_area=800,
    hitbox_size=40,
    start_time=25,
    kernel_size=15,
    blur_size=9,
    # Tracker MOG2+CSRT — ajustado para video con movimiento y efectos de luz:
    # mog_threshold: más alto que el default (30) para reducir falsos positivos.
    #   Sube a 50 si siguen apareciendo blobs de ruido; baja a 25 si se pierden detecciones.
    mog_threshold=35,
    # recording_lr: adaptación lenta del fondo durante grabación.
    #   Sube a 0.005 si la iluminación cambia rápido; baja a 0.001 si el ratón quieto desaparece.
    recording_lr=0.003,
)

input2 = MorrisPool(video_path= "../videos/MorrisPool", regions= RegionManager([
							CircularFractionRegion("H", (151, 110), 90, angle_start=270, fraction=0.25, overlap_threshold=0.75),
							]), 
							treatment="Escopolamina",
							subject_id="1",
							min_detection_area=100, 
							hitbox_size=10,
							start_time=5,
       						kernel_size=5,
    						blur_size=0
    						)

input3 = BarnesMaze(video_path= "../videos/BarnesMaze", regions= RegionManager([
                            CircleRegion("H", (320, 240), 40, overlap_threshold=0.75),
                            CircleRegion("N", (320, 60), 30, overlap_threshold=0.75),
                            CircleRegion("E", (540, 240), 30, overlap_threshold=0.75),
                            CircleRegion("S", (320, 420), 30, overlap_threshold=0.75),
                            CircleRegion("W", (100, 240), 30, overlap_threshold=0.75),
                            ]),
                            treatment="Control",
                            subject_id="1",
                            min_detection_area=2,
                            hitbox_size=30,
                            start_time=17.5,
                            kernel_size=15,
                            blur_size=15,
							#mog_threshold=30,
                            #recording_lr=0.001,
                            )