import time

import comet_ml

from load_data import load_official, load_labeled_test_data, load_and_evaluate_a_submission_file, \
    create_a_submission_file
from models.predefined_words import predefined_locations_predict

EXP_NAME: str = "tmp123123"  # "predefined_locations_predict_v4_threshold=5"

if EXP_NAME:
    print(f"Experiment name: {EXP_NAME}")
    time.sleep(3)  # time to cancel if no need to save results
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
    def predict(text):
        # model which based only on known locations
        return predefined_locations_predict(text, threshold=5)


    ### Create a submission file (and enhanced file with true locations)
    submission_file_path = f'out/{exp_name}_submission.csv'
    create_a_submission_file(official_test_data, predict, submission_file_path)

    ### Eval base on true location test set
    load_and_evaluate_a_submission_file(submission_file_path, exp)

# Some other Links:
# Evalution with https://github.com/rsuwaileh/seqeval
# Sklearn like model - https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html


exp.end()
