import argparse

import comet_ml

from load_data import load_official, load_labeled_test_data, load_and_evaluate_a_submission_file, \
    create_a_submission_file
from models.model import NERPipeline
from models.predefined_words import predefined_locations_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--disable", dest='disable', default=False, action='store_true')
    parser.add_argument("--eval_path", type=str, required=False, help="Experiment name")

    args = parser.parse_args()

    exp_name = args.name
    disabled = args.disable
    eval_submission_file_path = args.eval_path
    print(f"{exp_name=}")
    print(f"{disabled=}")

    if disabled:
        print(f"comet_ml is disabled")
    else:
        print(f"comet_ml is enabled with {exp_name=}")

    exp = comet_ml.Experiment(project_name="lmrc",
                              api_key="UfLYtYKuUTMlUsAzvWDbTT75k",
                              disabled=disabled)
    exp.set_name(exp_name)
    exp_name = exp.get_name()

    if eval_submission_file_path:
        # Only evaluate the submission file
        load_and_evaluate_a_submission_file(eval_submission_file_path, exp)
        print("Finish evaluation")
        exp.end()
        exit(0)

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


    ### Evaluate the predictor
    ner = NERPipeline()
    ### Create a submission file (and enhanced file with true locations)
    submission_file_path = f'out/{exp_name}_submission.csv'
    create_a_submission_file(official_test_data, ner.predict, submission_file_path, exp)

    ### Eval base on true location test set
    load_and_evaluate_a_submission_file(submission_file_path, exp)

    exp.end()

# Some other Links:
# Evalution with https://github.com/rsuwaileh/seqeval
# Sklearn like model - https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
