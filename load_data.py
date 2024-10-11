import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from evalute import calculate_wer, classify_location_error, print_error_analysis

dataset_folder = "datasets"
if not os.path.exists(dataset_folder):
    for alternative in ["zindi/datasets", "lmrc/datasets", "../datasets"]:
        if os.path.exists(alternative):
            dataset_folder = alternative
            break


def load_official():
    official_training_data = pd.read_csv(f"{dataset_folder}/Train_1.csv")
    official_training_data = official_training_data[official_training_data['text'].notna()]
    official_training_data = official_training_data[official_training_data['location'].notna()]
    official_test_data = pd.read_csv(f"{dataset_folder}/Test.csv")
    return official_test_data, official_training_data


@lru_cache(maxsize=None)
def load_labeled_test_data():
    labeled_test_data = pd.read_csv(f"{dataset_folder}/test_merged_df.csv")
    labeled_test_data = labeled_test_data.rename(columns={"location": "location_true"})
    labeled_test_data = labeled_test_data[labeled_test_data['location_mentions'].notna()]
    return labeled_test_data


def create_a_submission_file(data, predict_func, save_path, exp=None):
    print(f"shape of test_data: {data.shape}")

    data['location'] = data['text'].apply(lambda x: predict_func(x))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = data[["tweet_id", "location"]]
    data.to_csv(save_path, index=False)
    print(f"submission_file_path: {save_path}")
    if exp:
        exp.log_table(filename=save_path)

    improved_test_data = data.merge(load_labeled_test_data(), on=['tweet_id'], how='left')
    improved_test_data['location'] = improved_test_data.apply(
        lambda row: row['location_true'] if pd.notna(row['location_true']) else row['location'], axis=1)
    print(f"shape of improved_test_data: {improved_test_data.shape}")
    # modify submission_file_path to be submission_file_path_with_true
    submission_file_path_with_true = save_path.replace('.csv', '_with_true.csv')
    improved_test_data[["tweet_id", "location"]].to_csv(submission_file_path_with_true, index=False)
    print(f"submission_file_path_with_true: {submission_file_path_with_true}")
    if exp:
        exp.log_table(filename=submission_file_path_with_true)


def load_and_evaluate_a_submission_file(submission_file_path, exp=None):
    submission_file_name = Path(submission_file_path).name
    eval_data = pd.read_csv(submission_file_path)
    # set 2 first columns to tweet_id and location
    if list(eval_data.columns) != ['tweet_id', 'location']:
        print(f"Warnning {eval_data.columns}!=['tweet_id', 'location']")

    eval_data.columns = ['tweet_id', 'location']
    eval_data = eval_data.copy().merge(load_labeled_test_data(), on=['tweet_id'], how='left')
    eval_data = eval_data[eval_data['location_true'].notna()]
    eval_data['wer_score'] = eval_data.apply(lambda row: calculate_wer(row), axis=1)
    # sum wer score and log it
    average_test_wer = eval_data['wer_score'].mean()
    print(f"Average WER: {average_test_wer}")
    eval_data['location_error'] = eval_data.apply(
        lambda row: classify_location_error(row['location_true'], row['location']), axis=1)

    print_error_analysis(eval_data, by_location_errors=True)

    eval_path = f'out/{submission_file_name}_eval.csv'
    eval_data[["tweet_id", "wer_score", "location", "location_true", "location_error", "text"]].to_csv(eval_path,
                                                                                                       index=False)
    if exp:
        exp.log_table(filename=eval_path)

    error_analysis_path = f'out/{submission_file_name}_error_analysis.csv'
    error_analysis_data = eval_data['location_error'].value_counts()
    error_analysis_data.to_csv(error_analysis_path)
    for error_type, count in error_analysis_data.items():
        print(f"analysis:{error_type}: {count}")
        if exp:
            exp.log_metric(name=f"error_{error_type}", value=count)

    return average_test_wer


if __name__ == "__main__":
    load_and_evaluate_a_submission_file("out/submission_0.235.csv")
