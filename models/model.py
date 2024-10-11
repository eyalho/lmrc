import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from models.utils import capitalize_hashtag_words, extract_ner_names

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using device:", device)
else:
    device = torch.device("cpu")
    print('--------------------------------------------------')
    print("Warning: GPU is not available. Using device:", device)
    print("HuggingFace models may run significantly slower on CPU.")
    print('--------------------------------------------------')

model_name = "dslim/bert-base-NER"
model_name = "rsuwaileh/IDRISI-LMR-EN-random-typeless"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# Aggregation_strategy
Aggregation_strategy = "average"  # "average"?

ner_pipeline = pipeline("ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy=Aggregation_strategy,
                        device=device)

LOC_SCORE_TRESHOLD = 0.3
NAMES_LOW_SCORE = 0.7


def extract_locations_text(text, score_treshold=LOC_SCORE_TRESHOLD):
    ner_results = ner_pipeline(text)
    locations = [entity['word'] for entity in ner_results if
                 'LOC' in entity['entity_group']
                 and entity['score'] > score_treshold]

    if not locations:
        print(f"\n----\nwarning: no location in {text}\n{ner_results=}")

        # hastag_locations
        new_text = capitalize_hashtag_words(text)
        ner_results = ner_pipeline(new_text)
        print(f"{ner_results=}, {new_text=} {text=}")
        hastag_locations = extract_ner_names(text, ner_results, only_locations=True)
        if hastag_locations:
            print(f"capitalize_hashtag_words: {hastag_locations}")
            return hastag_locations

        # locations_low_score
        locations_low_score = [entity['word'] for entity in ner_results if
                               'LOC' in entity['entity_group']
                               and entity['score'] <= score_treshold]
        if locations_low_score:
            print(f"return:{locations_low_score=}")
            return locations_low_score

        # names_low_score_not_locations
        names_low_score_not_locations = [entity['word'] for entity in ner_results if
                                         entity['score'] < NAMES_LOW_SCORE]
        if names_low_score_not_locations:
            print(f"return:{names_low_score_not_locations=}")
            return names_low_score_not_locations

    return sorted(set(locations))





if __name__ == "__main__":
    example = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    extract_locations_text(example)
    print(f"{example}")





    # def apply_ner_pipeline(extract_locations_func):
    #     data = {
    #         "text": [
    #             "I live in New York but I often travel to Paris and London.",
    #             "The Eiffel Tower is in Paris, France.",
    #             "Tokyo is the capital of Japan.",
    #             "Irma death toll in US climbs to 12 as part of Florida Keys reopen to residents - @ABC News - #HurricaneIrma #Florida",
    #             # [Florida Florida US]
    #             "Newsweek: Paradise lost: Inside the burned-out California town destroyed by deadly Camp Fire.  via @GoogleNews",
    #             # [California]
    #             "RT @grey_fortress_: #CycloneIdai donations from Zimbabwe Catholics in Johannesburg.",
    #             # [Johannesburg Zimbabwe]
    #             "Incredible scenes of destruction in Beira, Mozambique under the effects of Tropical Cyclone Idai. Report via Meteo Tras Os Montes.",
    #             # [Beira Mozambique]
    #             "RT @ScoopNZ: Magnitude 7.5 Earthquake strikes near Hamner in North Canterbury. Damage reported over a wide area. #EQNZ",
    #             # [Canterbury]
    #             ""
    #
    #         ]
    #     }
    #     small_test_validation_df = pd.DataFrame(data)
    #     small_test_validation_df['location_predict'] = small_test_validation_df['text'].apply(extract_locations_func)
    #     print('----')
    #     for index, row in small_test_validation_df.iterrows():
    #         print(row['text'], row['location_predict'])
    #     print('----')
    #
    # apply_ner_pipeline(extract_locations_text)
    # apply_ner_pipeline(ner_pipeline)
    #
    #
