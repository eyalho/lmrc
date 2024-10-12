from typing import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from models.utils import capitalize_hashtag_words, extract_ner_names, fix_locations, capitalize_known_words

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using device:", device)
else:
    device = torch.device("cpu")
    print('--------------------------------------------------')
    print("Warning: GPU is not available. Using device:", device)
    print("HuggingFace models may run significantly slower on CPU.")
    print('--------------------------------------------------')


class NERPipeline:
    def __init__(self, config: dict):
        self.config = config  # contains model_name and other hyperparameters
        model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.Aggregation_strategy = "average"  # "average"?
        self.ner_pipeline = pipeline("ner",
                                     model=self.model,
                                     tokenizer=self.tokenizer,
                                     aggregation_strategy=self.Aggregation_strategy,
                                     device=device)
        # self.post_blacklist_names = load_post_blacklist_names()

    def preprocess_on_second_try(self, text):
        text = text.replace('-', ' ').replace('/', ' ')
        return text

    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s\./\-_]', '', text)

        if self.config.get('capitalize_hashtag'):
            text = capitalize_hashtag_words(text)

        if self.config.get('capitalize_known_words'):
            text = capitalize_known_words(text)
        return text

    def postprocess(self, text, ner_results):
        locations_list = extract_ner_names(text, ner_results, only_locations=True,
                                           merge_locations=self.config.get('merge_locations'))
        locations_list = sorted(set(locations_list))
        if self.config.get('fix_locations'):
            locations_list = fix_locations(locations_list, text)
        return locations_list

    def predict(self, text):
        original_text = text
        process_text = self.preprocess(text)
        ner_results = self.ner_pipeline(process_text)
        return self.postprocess(text, ner_results)


if __name__ == "__main__":
    example = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    print(f"{example}")
    print(f"{NERPipeline().predict(example)=}")

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
