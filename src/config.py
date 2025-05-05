import os

BASE_DIR    = os.path.abspath(os.path.dirname(__file__) + "/../")
DATA_DIR    = os.path.join(BASE_DIR, "data")
DATA_PATH   = os.path.join(DATA_DIR, "tweets_politica.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Asegura que existan
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
