import sys
from src.runner import main as run_experiments

if __name__ == "__main__":
    # Delegate to the experiments CLI if invoked directly:
    # Examples:
    #   python main.py --suite math --mode shot --cap 32 --shots_list 0 2 6 8 10 12 14 --plot
    #   python main.py --suite sentiment --mode cap --shots 2 --caps_list 32 64 128 256 512 --plot
    sys.exit(run_experiments())