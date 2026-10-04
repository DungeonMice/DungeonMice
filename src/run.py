import sys
from input_loader import load_experiment
from RunExperiment import main


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit(1)

	json_path = sys.argv[1]

	try:
		labyrinth = load_experiment(json_path)
	except (FileNotFoundError, ValueError) as e:
		print(f"[ERROR] {e}")
		sys.exit(1)

	main(labyrinth)