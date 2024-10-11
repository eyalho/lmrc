from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from models.utils import capitalize_hashtag_words, extract_ner_names, fix_locations

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using device:", device)
else:
    device = torch.device("cpu")
    print('--------------------------------------------------')
    print("Warning: GPU is not available. Using device:", device)
    print("HuggingFace models may run significantly slower on CPU.")
    print('--------------------------------------------------')


@lru_cache(maxsize=None)
def func_ner_pipeline(model_name="dslim/bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    Aggregation_strategy = "average"  # "average"?
    ner_pipeline = pipeline("ner",
                            model=model,
                            tokenizer=tokenizer,
                            aggregation_strategy=Aggregation_strategy,
                            device=device)
    return ner_pipeline

def simple_ner(text, model_name="rsuwaileh/IDRISI-LMR-EN-random-typeless"):
    # model_name = "dslim/bert-base-NER"
    ner_results = func_ner_pipeline(text)
    return  ner_results


def simple_ner_predict(text):
    simple_ner_results = simple_ner(text)
    locations_list = extract_ner_names(text, simple_ner_results)
    locations_list = sorted(set(locations_list))
    return fix_locations(locations_list, text)


if __name__ == "__main__":
    example = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    print(f"{example}")
    print(f"{simple_ner(example)=}")

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
