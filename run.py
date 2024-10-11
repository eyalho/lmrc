import os

import comet_ml
import pandas as pd

from evalute import print_error_analysis, classify_location_error, calculate_wer
from load_data import load_official, load_labeled_test_data
from models.utils import extract_ner_words, extract_predefined_locations

EXP_NAME: str = "predefined_locations_predict_v1"

if EXP_NAME:
    print(f"Experiment name: {EXP_NAME}")
    disabled = False
else:
    print(f"comet_ml is disabled")
    disabled = True
exp = comet_ml.Experiment(project_name="lmrc",
                          api_key="UfLYtYKuUTMlUsAzvWDbTT75k",
                          disabled=disabled)
exp.set_name(EXP_NAME)
exp_name = exp.get_name()
assert exp_name == EXP_NAME

if __name__ == "__main__":

    ### load data
    official_test_data, official_training_data = load_official()
    labeled_test_data = load_labeled_test_data()
    print(f"{official_training_data.shape=}")
    print(f"{official_test_data.shape=}")
    print(f"{labeled_test_data.shape=}")


    ### Choose a predictor
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


    ### Create a submission file
    official_test_data['location'] = official_test_data['text'].apply(lambda x: predefined_locations_predict(x))
    submission_file_path = f'out/{exp_name}_submission.csv'
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    official_test_data[["tweet_id", "location"]].to_csv(submission_file_path, index=False)

    #### Improve submission by true location if available
    improved_test_data = official_test_data.copy().merge(labeled_test_data, on=['tweet_id', 'text'], how='left')
    improved_test_data['location'] = improved_test_data.apply(
        lambda row: row['location_true'] if pd.notna(row['location_true']) else row['location'], axis=1)
    print(f"shape of test_data: {improved_test_data.shape}")
    # modify submission_file_path to be submission_file_path_with_true
    submission_file_path_with_true = submission_file_path.replace('.csv', '_with_true.csv')
    improved_test_data[["tweet_id", "location"]].to_csv(submission_file_path_with_true, index=False)

    ### Eval base on true location test set
    eval_data = official_test_data.copy().merge(labeled_test_data, on=['tweet_id', 'text'], how='left')
    eval_data = eval_data[eval_data['location_true'].notna()]
    print(f"shape of eval_data: {eval_data.shape}")

    # calculate WER score
    eval_data['wer_score'] = eval_data.apply(lambda row: calculate_wer(row), axis=1)
    # sum wer score and log it
    average_test_wer = eval_data['wer_score'].mean()
    print(f"Average WER: {average_test_wer}")
    exp.log_metric(name="test_wer", value=average_test_wer)

    # left join eval_data and labeled_test_data on [tweet_id, text] and add column location_true from  labeled_test_data
    print_error_analysis(eval_data)

    # apply classify_location_error on each row of eval_data and add column location_error
    eval_data['location_error'] = eval_data.apply(
        lambda row: classify_location_error(row['location_true'], row['location']), axis=1)

    eval_path = f'out/{exp_name}_eval.csv'
    eval_data[["tweet_id", "wer_score", "location", "location_true"]].to_csv(eval_path, index=False)
    exp.log_table(filename=eval_path)

    error_analysis_path = f'out/{exp_name}_error_analysis.csv'
    error_analysis_data = eval_data['location_error'].value_counts()
    error_analysis_data.to_csv(error_analysis_path)
    exp.log_table(filename=error_analysis_path)
    for error_type, count in error_analysis_data.items():
        print(f"analysis:{error_type}: {count}")
        exp.log_metric(name=f"error_{error_type}", value=count)

# Some other Links:
# Evalution with https://github.com/rsuwaileh/seqeval
# Sklearn like model - https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html


exp.end()
