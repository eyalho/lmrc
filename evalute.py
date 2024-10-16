from jiwer import wer


def calculate_wer(row):
    reference = row['location_true']
    hypothesis = row['location']
    return wer(reference, hypothesis)


def _print_error_analysis(val_df):
    error_analysis_df = val_df[val_df['location'] != val_df['location_true']]
    print(f"Number of errors: {len(error_analysis_df)} ({len(error_analysis_df) / len(val_df) * 100}%)")
    i = 0
    for _, row in error_analysis_df.iterrows():
        print("--")
        print("Text:", row['text'])
        print("True Location:", row['location_true'])
        print("Predicted Location:", row['location'])
        i += 1
        if i > 3:
            break


def print_error_analysis(val_df, by_location_errors=False):
    if not by_location_errors:
        return _print_error_analysis(val_df)

    error_analysis_data = val_df['location_error'].value_counts()
    for error_type, count in error_analysis_data.items():
        print('\n--------------------------------------------')
        print(f"analysis:{error_type}: {count}\n")
        _print_error_analysis(val_df[val_df['location_error'] == error_type].sample(frac=1))
    print('\n')


# Categorizing functions
def no_predicted_location(true_loc, pred_loc):
    """Check if no location was predicted."""
    return pred_loc == "no_locations_found"


def is_pred_subset_of_true(true_loc, pred_loc):
    """Check if the predicted location is a subset of the true location."""
    true_parts = set(true_loc.split())
    pred_parts = set(pred_loc.split())
    return pred_parts.issubset(true_parts)


def is_true_subset_of_pred(true_loc, pred_loc):
    """Check if the true location is a subset of the predicted location."""
    true_parts = set(true_loc.split())
    pred_parts = set(pred_loc.split())
    return true_parts.issubset(pred_parts)


def is_location_order_problem(true_loc, pred_loc):
    """Check if the prediction and true location have common parts but are not exactly the same."""
    true_parts = set(true_loc.split())
    pred_parts = set(pred_loc.split())
    return true_parts == pred_parts and true_loc != pred_loc


def is_location_confusion(true_loc, pred_loc):
    """Check if the prediction and true location have common parts but are not exactly the same."""
    true_parts = set(true_loc.split())
    pred_parts = set(pred_loc.split())
    return len(true_parts & pred_parts) > 0 and true_loc != pred_loc


def has_extraneous_info(pred_loc):  # TODO
    """Check if the prediction contains irrelevant or excessive information."""
    return len(pred_loc.split()) > 4


# Error classification
def classify_location_error(true_loc: str, pred_loc: str):
    """Classify the type of error based on true and predicted locations."""
    if true_loc == pred_loc:
        return "correct"
    if no_predicted_location(true_loc, pred_loc):
        return "no_location_found"
    if is_location_order_problem(true_loc, pred_loc):
        return "location_order_problem"
    elif is_pred_subset_of_true(true_loc, pred_loc):
        return "subset_of_true"
    elif is_true_subset_of_pred(true_loc, pred_loc):
        return "contains_true_but_not_equal"
    elif is_location_confusion(true_loc, pred_loc):
        return "location_confusion"
    elif has_extraneous_info(pred_loc):
        return "wrong_and_more_than_4_words"
    else:
        return "Unknown Error"

# if __name__ == "__main__":
#     from tqdm import tqdm
#     tqdm.pandas()
#     val_df['location_predict_list'] = val_df['text'].progress_apply(extract_locations_text)
#     val_df['location_predict'] = val_df['location_predict_list'].apply(
#         lambda x: ' '.join(x) if x else "no_locations_found")
#     val_df[['text', 'location', 'location_predict']].head()
#     val_df['wer_score'] = val_df.apply(calculate_wer, axis=1)
#
#     average_wer = val_df['wer_score'].mean()
#     print(f"Average WER: {average_wer}")
