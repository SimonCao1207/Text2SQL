import json

from tqdm import tqdm

from const import (
    EHR_TEST_DATA_PATH,
    EHR_TEST_LABEL_PATH,
)
from utils import (
    generate_classification_answer,
    get_tokenizer_model,
    load_data,
)


def main(is_save=False):
    model, tokenizer = get_tokenizer_model()
    test_data, test_label = load_data(EHR_TEST_DATA_PATH, EHR_TEST_LABEL_PATH)
    test_data = test_data["data"]

    total = 0
    n_yes = 0
    n_no = 0

    no_as_yes = 0
    no_as_no = 0
    yes_as_no = 0
    yes_as_yes = 0
    n_invalid = 0

    for sample in tqdm(test_data):
        id_ = sample["id"]
        question = sample["question"]

        total += 1
        if test_label[id_] == "null":
            n_no += 1
            answer = "NO"
            sample["answer"] = answer
        else:
            n_yes += 1
            answer = "YES"
            sample["answer"] = answer

        pred_answer = generate_classification_answer(question, model, tokenizer)

        if pred_answer == "YES" and answer == "YES":
            yes_as_yes += 1
        elif pred_answer == "NO" and answer == "NO":
            no_as_no += 1
        elif pred_answer == "NO" and answer == "YES":
            yes_as_no += 1
        elif pred_answer == "YES" and answer == "NO":
            no_as_yes += 1
        else:
            n_invalid += 1
            print(f"Generated answer: {pred_answer}")

    correct = no_as_no + yes_as_yes
    print(f"Total number of data: {total}")
    print(f"Number of YES: {n_yes}")
    print(f"Number of NO: {n_no}")

    print(
        f"Number of correct predictions and accuracy: {correct, 100 * correct / total}"
    )

    print(f"Number of NO as YES: {no_as_yes}")
    print(f"Number of YES as NO: {yes_as_no}")
    print(f"Number of YES as YES: {yes_as_yes}")
    print(f"Number of NO as NO: {no_as_no}")
    print(f"Number of INVALID predictions: {n_invalid}")

    if is_save:
        output_path = "classification_results.json"
        with open(output_path, "w") as f:
            json.dump(test_data, f)


if __name__ == "__main__":
    main()
