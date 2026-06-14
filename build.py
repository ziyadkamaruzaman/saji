import os
from src.saji.build_index import build_index

if __name__ == "__main__":
    DATA_DIR = "data/raw"
    ART_DIR = "artifacts"
    OV_PATH = "overrides/halal_overrides.json"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    
    print("Building Saji index...")
    build_index(DATA_DIR, ART_DIR, OV_PATH)
    print(" Index built successfully at artifacts/")