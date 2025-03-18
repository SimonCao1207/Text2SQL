from const import (
    EHR_TEST_DATA_PATH,
    EHR_TEST_LABEL_PATH,
)
from utils import get_scores, load_data, submit


def run_baseline():
    # Load data from ehr test dataset
    test_data, test_labels = load_data(EHR_TEST_DATA_PATH, EHR_TEST_LABEL_PATH)

    label_y = {idx: "null" for idx, sql in test_labels.items()}
    return test_data, test_labels, label_y


if __name__ == "__main__":
    test_data, test_labels, label_y = run_baseline()
    submit(label_y, "ehr_test_null.json")
    get_scores(test_data, test_labels, label_y)
