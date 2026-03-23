from labyrinths.labyrinth_MorrisPool import MorrisPool
from labyrinths.labyrinth_CrossMaze import CrossMaze
from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion

input1 = CrossMaze(
    video_path="../videos/CrossMaze",
    treatment="Control",
    subject_id="1",
    regions=RegionManager([
        PolygonRegion("este",  [[630,420],[893,420],[893,350],[630,350]], overlap_threshold=0.75),
        PolygonRegion("oeste", [[288,420],[288,350],[550,350],[550,420]], overlap_threshold=0.75),
        PolygonRegion("norte", [[555,350],[625,350],[625,87],[555,87]], overlap_threshold=0.75),
        PolygonRegion("sur", [[555,683],[625,683],[625,420],[555,420]], overlap_threshold=0.75)
    ]),
    min_detection_area=2000,
    hitbox_size=40,
    start_time=25,
    kernel_size=15,
    blur_size=9 # debe ser impar: 7, 11, 15, 21
)

input2 = {
	"video_path" : "../Escopolamina 1.avi",
	"regions": RegionManager([
							CircleRegion("centro",  [151,110], 40),
							]),
	
}

input3 = {
	"video_path" : "../Escopolamina 1.avi",
	"regions": RegionManager([
							CircularFractionRegion("H", (151, 110), 50, angle_start=90, fraction=0.5),
							]),
	
}

input4 = MorrisPool(video_path= "../videos/MorrisPool", regions= RegionManager([
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