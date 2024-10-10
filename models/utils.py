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


def extract_predefined_locations(text):
    from models.predefined_words import PREDEFINED_LOCATIONS
    extracted_ner_output = []
    for predefined_location in PREDEFINED_LOCATIONS:
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


if __name__ == "__main__":
    text = "Aftershocks expected in earthquake-hit areas within 24 hours: NDMA. #pakistan"
    new_text = capitalize_hashtag_words(text)
    print(f"{text} -> {new_text}")
    ner_results = [{'entity_group': 'ORG', 'score': 0.6138262, 'word': 'NDMA', 'start': 62, 'end': 66},
                   {'entity_group': 'LOC', 'score': 0.99970454, 'word': 'Pakistan', 'start': 69, 'end': 77}]
    print(f"{extract_ner_words(text, ner_results, only_locations=False)=}")
    print(f"{extract_ner_words(text, ner_results, only_locations=True)=}")
    print(f"{extract_ner_words(text, extract_predefined_locations(text))=}")
