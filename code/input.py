from labyrinth import MorrisPool, CrossMaze 
from regions import RegionManager, PolygonRegion, CircleRegion, CircularFractionRegion

input1 = CrossMaze(
    video_path="../Mice maze experiment.mp4",
    treatment="Control",
    subject_id="1",
    regions=RegionManager([
        PolygonRegion("este",  [[620,450],[903,450],[900,320],[622,320]]),
        PolygonRegion("oeste", [[272,450],[274,320],[566,320],[562,450]])
    ]),
    min_detection_area=2000,
    hitbox_size=40,
    start_time=5
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

input4 = MorrisPool(video_path= "../Escopolamina 1.avi", regions= RegionManager([
							CircularFractionRegion("H", (151, 110), 80, angle_start=270, fraction=0.25),
							]), 
							treatment="Escopolamina",
							subject_id="1",
							min_detection_area=100, 
							hitbox_size=10, 
							start_time=5)