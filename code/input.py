from regions import RegionManager, PolygonRegion, CircleRegion

input1 = {
	"video_path" : "Mice maze experiment.mp4",
	"regions": RegionManager([
							PolygonRegion("este",  [[620,450],[903,450],[900,320],[622,320]]),
							PolygonRegion("oeste", [[272,450],[274,320],[566,320],[562,450]])
							]),
	
}

input2 = {
	"video_path" : "Escopolamina 1.avi",
	"regions": RegionManager([
							CircleRegion("centro",  [151,110], 40),
							]),
	
}

input3 = {
	"video_path" : "Escopolamina 1_1280x720.avi",
	"regions": RegionManager([
							CircleRegion("centro",  [617,330], 100),
							]),
	
}