from collections import Counter
from functools import lru_cache

from load_data import load_official
from models.utils import extract_ner_names, fix_locations


@lru_cache(maxsize=None)
def _read_locations(file_path, threshold):
    import pandas as pd
    df = pd.read_csv(file_path, header=0)
    df['count'] = df['count'].astype(int)
    return {row['location'] for _, row in df.iterrows() if row['count'] > threshold}


def read_predefined_locations(threshold=5):
    C = Counter(load_official()[1]['location'].values)
    locations = {c for c in C if C[c] > threshold}
    locations = locations.union({'WalMart', 'MONTANA', 'Andrapradesh', 'Zimbabwes', 'Greeces', 'Creek', 'Kugaaruk', 'Thimbirigasyaya', 'Palestinian', 'florida', 'NC', 'Osborn', 'county', 'Camp', 'Center', 'brooklyn', 'Nebraska', 'Gainesville', 'Harare', 'Floridas', 'Village', 'St', 'France', 'Norm', 'Islands', 'Via', 'Island', 'Lowcountry', 'TamilNadu', 'Jaspe', 'Miami', 'BNZ', 'FortStJohn', 'Nepal', 'CDMX', 'Zimutos', 'JK', 'US', 'CHRISTCHURCH', 'Bay', 'Nadakkavu', 'Cross', 'Sunrise', 'HAMPSTEAD', 'J&K', 'Orlando', 'FortMcmurray', 'Chico’s', 'Chipinge', 'Rico', 'USVirginIslands', 'Red', 'inland', 'Kingwood', 'Carolina', 'Ecuadors', 'Fla.', 'earthquake', 'NE', 'YMM', 'Gerald', 'Anzac', 'Pedernales', 'Mar-A-Lago', '9/11,California', 'Amartice', 'cost', 'Haitis', 'HOLLYWOOD', 'building', 'Parnassus', 'Florence', 'Timaru', 'California', 'Victory', 'Hai', 'TEXAS', 'the', 'Californias', 'Midwestern', 'Rockport', 'COUNTY', 'Macerata', 'UKAid', 'Amatrice', 'FL', 'godavari', 'Amrutha', 'Britomart', 'Indian', 'Area', 'Ocala', 'Floridians', 'HUMBOLDT', 'vidyalayam', 'Mexicos', 'Hamont', 'Keralite', 'Cochin', 'Keralas', 'Mati', '3', 'Dade', 'Tsipras', 'Dom', 'Melb', 'AbacoIsland', 'Edappally', 'Nsanje', 'Italian', 'NZs', 'Italy', 'County', 'StThomas', 'Haiti-', 'Chch', 'Mexico', 'Highway', 'Iowans', 'Miamis', 'CycloneIdai', 'East', 'Salaria', 'Coast', 'FortMcMurray', 'State', 'Nebraskas', 'zimbabwe', 'FortMacFire', 'road', 'Haiti', 'Republic', 'LUCIE', 'Rafina', 'Rescue', 'Bank', 'Greece', 'UAE', 'jallas', 'Amatrices', 'Pak', 'DC', 'DUBAI', 'Carolinas', 'Seddon', 'wellingtonnz', 'PONCE', 'Dorian', 'FLL', 'Volusia', 'Emergency', 'EcuadorEarthquake', 'Syrian', 'Puerto', 'MX', 'Marche', 'Meanwhile', 'GREECE', 'North', 'DominicanRepublic', 'Amitrice', 'coast', 'Operations', 'Hampstead', 'MacDonald', 'Canadas', 'SAINT', 'lakeland', 'Black', 'PENDER', 'Bourke', 'airport', 'Columbia', 'Vincent', 'Charlestons', 'Canadian', 'Vimbai', 'Aluva'})
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
    print(f"{extracted_ner_output=}")
    return extracted_ner_output


def predefined_locations_predict(text, threshold=5):
    locations_list = extract_ner_names(text, extract_predefined_locations(text, threshold=threshold))
    locations_list = sorted(set(locations_list))
    return fix_locations(locations_list, text)


if __name__ == "__main__":
    print(f"{len(read_predefined_locations(threshold=5))} locations found.")
    text = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    print(f"{extract_ner_names(text, extract_predefined_locations(text))=}")
    text = "What is happening to the infrastructure in New England? It isnt global warming, its misappropriated funds being abused that shouldve been used maintaining their infrastructure that couldve protected them from floods! Like New Orleans. Their mayor went to ὄ7#Maryland #floods"
    print(f"{extract_ner_names(text, extract_predefined_locations(text))=}")
    print(f"[{predefined_locations_predict(text, threshold=10)}], vs [New England New Orleans]")

    text = 'RT @BJP4Andhra: Central assistance to Kerala Floods relief - proactive, rapid & multi-modal. #PMModiWithKeralam #KeralaFloods /'
    print(f"{extract_ner_names(text, extract_predefined_locations(text))=}")
    print(f"[{predefined_locations_predict(text, threshold=1)}], vs [Kerala]")
