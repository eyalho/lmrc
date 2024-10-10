import csv


def _read_locations(file_path, threshold):
    import pandas as pd
    df = pd.read_csv(file_path, header=0)
    df['count'] = df['count'].astype(int)
    return {row['location'] for _, row in df.iterrows() if row['count'] > threshold}

def read_locations(file_path='datasets/location_counts.csv', threshold=5):
    try:
        locations = _read_locations(file_path, threshold)
    except FileNotFoundError:
        locations = _read_locations("../" + file_path, threshold)
    return locations


PREDEFINED_LOCATIONS = read_locations(threshold=5)

if __name__ == "__main__":
    print(f"{len(read_locations())} locations found.")
