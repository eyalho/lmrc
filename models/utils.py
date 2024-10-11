import re
from typing import List

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

# nltk.data.path.append('/usr/share/nltk_data/')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')


def capitalize_hashtag_words(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.startswith("#"):
            new_words.append(word[0] + word[1:].capitalize())
        else:
            new_words.append(word)
    return " ".join(new_words)


def extract_ner_words(text, ner_results, only_locations=True):
    # Create a list to hold the extracted words
    # This function is based on start, end positions of the words
    # So it catch the right form (capitalized or not) of the word
    extracted_words = []

    # Iterate through the NER results
    for result in ner_results:
        start = result['start']
        end = result['end']
        if only_locations:
            if 'LOC' in result['entity_group']:
                extracted_words.append(text[start:end])
            else:
                continue
        else:
            extracted_words.append(text[start:end])

    return extracted_words


def filter_locations_by_word(locations_list: List[str], text: str):
    # # todo fix
    # extract_ner_words(text, extract_predefined_locations(text))=['England', 'Maryland', 'New Orleans']
    # [Unknown], vs [New England New Orleans]
    text_words = preprocess_text(text)
    locations_list = [location for location in locations_list if location.split(' ')[0] in text_words]
    return locations_list


def fix_locations(locations_list: List[str], text: str):
    locations_list = filter_locations_by_word(locations_list, text)

    sublocations_list = ['New', 'new', 'Ellicott']
    for location in locations_list:
        for sublocation in locations_list:
            if sublocation in location and location != sublocation and sublocation:
                if sublocation in sublocations_list:
                    locations_list.remove(sublocation)
                else:
                    try: #todo fix logic?
                        locations_list.remove(location)  # remove the longer one
                    except ValueError:
                        pass


    if not locations_list:
        locations_list = ["Unknown"]  # assuming all texts have at least one location
    locations_list = sorted(set(locations_list))
    return " ".join(locations_list)


def preprocess_text(text) -> list[str]:
    # Remove URLs
    # text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
    # # Remove user mentions
    # text = re.sub(r'@\w+', '', text)
    # # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z0-9\s\./\-_]', '', text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    #     # Lemmatize
    #     lemmatizer = WordNetLemmatizer()
    #     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return words


if __name__ == "__main__":
    text = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    new_text = capitalize_hashtag_words(text)
    print(f"{text} -> {new_text}")
    ner_results = [{'entity_group': 'ORG', 'score': 0.6138262, 'word': 'NDMA', 'start': 62, 'end': 66},
                   {'entity_group': 'LOC', 'score': 0.99970454, 'word': 'Pakistan', 'start': 69, 'end': 77}]
    print(f"{extract_ner_words(text, ner_results, only_locations=False)=}")
    print(f"{extract_ner_words(text, ner_results, only_locations=True)=}")

    print(f"{preprocess_text(text)=}")
    print(f'{" ".join(preprocess_text(text))=}')
