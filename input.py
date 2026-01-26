from regions import RegionManager, Region

input1 = {
	"video_path" : "Mice maze experiment.mp4",
	"regions": RegionManager([
							Region("este",  [[620,450],[903,450],[900,320],[622,320]]),
							Region("oeste", [[272,450],[274,320],[566,320],[562,450]])
							]),
	
}

input2 = {
	"video_path" : "Escopolamina 1.avi",
	"regions": RegionManager([
							Region("este",  [[620,450],[903,450],[900,320],[622,320]]),
							Region("oeste", [[272,450],[274,320],[566,320],[562,450]])
							]),
	
}