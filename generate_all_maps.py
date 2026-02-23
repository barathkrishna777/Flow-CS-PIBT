import os
import glob
import numpy as np

MAP_DIR = "data/mapf-map"
OUTPUT_NPZ = "data/all_maps.npz"

def read_map(map_file):
    """Reads the .map file and returns a 2D numpy array (0 for empty, 1 for obstacle)."""
    with open(map_file, 'r') as f:
        f.readline() # type octile
        height = int(f.readline().split()[1])
        width = int(f.readline().split()[1])
        f.readline() # map
        
        map_data = np.zeros((height, width), dtype=int)
        for r in range(height):
            line = f.readline().strip()
            for c in range(width):
                if line[c] in ['@', 'T', 'O', 'W']:
                    map_data[r, c] = 1
    return map_data

def main():
    map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
    map_dict = {}
    
    print(f"Bundling {len(map_files)} maps...")
    for map_path in map_files:
        map_name = os.path.basename(map_path) # e.g., 'random-32-32-20.map'
        map_dict[map_name] = read_map(map_path)
        
    np.savez_compressed(OUTPUT_NPZ, **map_dict)
    print(f"Successfully saved to {OUTPUT_NPZ}!")

if __name__ == "__main__":
    main()