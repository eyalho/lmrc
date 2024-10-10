import pandas as pd

dataset_folder = "datasets"


def load_official():
    official_training_data = pd.read_csv(f"{dataset_folder}/Train_1.csv")
    official_training_data = official_training_data[official_training_data['text'].notna()]
    official_training_data = official_training_data[official_training_data['location'].notna()]
    official_test_data = pd.read_csv(f"{dataset_folder}/Test.csv")
    return official_test_data, official_training_data


def load_labeled_test_data():
    labeled_test_data = pd.read_csv(f"{dataset_folder}/test_merged_df.csv")
    labeled_test_data = labeled_test_data.rename(columns={"location": "location_true"})
    labeled_test_data = labeled_test_data[labeled_test_data['location_mentions'].notna()]
    return labeled_test_data
