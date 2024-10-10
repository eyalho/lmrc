import comet_ml

from evalute import print_error_analysis, classify_location_error, calculate_wer
from load_data import load_official, load_labeled_test_data
from models.utils import extract_ner_words, extract_predefined_locations

exp = None
EXP_NAME: str | None = "predefined_locations_predict_v1"

if EXP_NAME:
    ENABLE_COMMET_ML = True
else:
    ENABLE_COMMET_ML = False

if ENABLE_COMMET_ML:
    exp = comet_ml.Experiment(project_name="lmrc", api_key="UfLYtYKuUTMlUsAzvWDbTT75k")
    if EXP_NAME:
        exp.set_name(EXP_NAME)

    exp.log_code()

if __name__ == "__main__":
    official_test_data, official_training_data = load_official()
    labeled_test_data = load_labeled_test_data()
    print(f"{official_training_data.shape=}")
    print(f"{official_test_data.shape=}")
    print(f"{labeled_test_data.shape=}")


    # model which based only on known locations:

    def predefined_locations_predict(text):
        # Wrong example
        # C2,C3
        # England Maryland New Orleans,New England New Orleans
        locations_list = extract_ner_words(text, extract_predefined_locations(text))
        locations_list = sorted(set(locations_list))
        # also if sublocation is in the list, remove it
        for location in locations_list:
            for sublocation in locations_list:
                if sublocation in location and location != sublocation:
                    locations_list.remove(sublocation)

        if not locations_list:
            locations_list = ["Unknown"]  # assuming all texts have at least one location
        return " ".join(locations_list)


    official_test_data['location'] = official_test_data['text'].apply(lambda x: predefined_locations_predict(x))

    official_test_data[["tweet_id", "location"]].to_csv('out/eval_predefined_locations_predict.csv', index=False)

    test_data = official_test_data.copy().merge(labeled_test_data, on=['tweet_id', 'text'], how='left')
    print(f"shape of test_data: {test_data.shape}")
    eval_data = test_data.copy()[test_data['location_true'].notna()]
    print(f"shape of eval_data: {eval_data.shape}")

    # calculate WER score
    eval_data['wer_score'] = eval_data.apply(lambda row: calculate_wer(row), axis=1)
    # sum wer score and log it
    average_test_wer = eval_data['wer_score'].mean()
    print(f"Average WER: {average_test_wer}")
    if exp:
        exp.log_metric(name="test_wer", value=average_test_wer)

    eval_data[["tweet_id", "wer_score", "location", "location_true"]].to_csv(
        'out/eval_predefined_locations_predict.csv', index=False)

    # left join eval_data and labeled_test_data on [tweet_id, text] and add column location_true from  labeled_test_data
    print_error_analysis(eval_data)

    # apply classify_location_error on each row of eval_data and add column location_error
    eval_data['location_error'] = eval_data.apply(
        lambda row: classify_location_error(row['location_true'], row['location']), axis=1)

    print(eval_data['location_error'].value_counts())

    eval_data.to_csv('out/eval.csv', index=False)

    if exp:
        exp.log_table(filename="out/eval.csv")

# Some other Links:
# Evalution with https://github.com/rsuwaileh/seqeval
# Sklearn like model - https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
