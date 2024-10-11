from functools import lru_cache
from pathlib import Path

from models.utils import extract_ner_words, fix_locations


@lru_cache(maxsize=None)
def _read_locations(file_path, threshold):
    import pandas as pd
    df = pd.read_csv(file_path, header=0)
    df['count'] = df['count'].astype(int)
    return {row['location'] for _, row in df.iterrows() if row['count'] > threshold}


def read_predefined_locations(threshold=5):
    # get path of current file
    file_path = str(Path(__file__).parent.parent / 'datasets/location_counts.csv')
    locations = _read_locations(file_path, threshold)
    return locations


def extract_predefined_locations(text, threshold=5):
    extracted_ner_output = []

    for predefined_location in read_predefined_locations(threshold=threshold):
        start_idx = text.find(predefined_location)
        if start_idx != -1:
            end_idx = start_idx + len(predefined_location)
            extracted_ner_output.append({
                'entity_group': 'LOC',
                'score': 0.5,
                'word': predefined_location,
                'start': start_idx,
                'end': end_idx
            })
    return extracted_ner_output








def predefined_locations_predict(text, threshold=5):
    locations_list = extract_ner_words(text, extract_predefined_locations(text, threshold=threshold))
    locations_list = sorted(set(locations_list))
    return fix_locations(locations_list, text)


if __name__ == "__main__":
    print(f"{len(read_predefined_locations(threshold=5))} locations found.")
    text = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    print(f"{extract_ner_words(text, extract_predefined_locations(text))=}")
    text = "What is happening to the infrastructure in New England? It isnt global warming, its misappropriated funds being abused that shouldve been used maintaining their infrastructure that couldve protected them from floods! Like New Orleans. Their mayor went to ὄ7#Maryland #floods"
    print(f"{extract_ner_words(text, extract_predefined_locations(text))=}")
    print(f"[{predefined_locations_predict(text, threshold=10)}], vs [New England New Orleans]")

    text = 'RT @BJP4Andhra: Central assistance to Kerala Floods relief - proactive, rapid & multi-modal. #PMModiWithKeralam #KeralaFloods /'
    print(f"{extract_ner_words(text, extract_predefined_locations(text))=}")
    print(f"[{predefined_locations_predict(text, threshold=1)}], vs [Kerala]")
