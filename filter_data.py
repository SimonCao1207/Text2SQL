import json
import pickle
import re

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from const import RESERVED_WORDS
from scoring.utils import (
    execute_sql_for_evaluator,
    postprocess_gt,
    postprocess_pred,
    process_answer,
)

with open("results/ehr_test_baseline_preds.json", "r") as f:
    baseline_preds = json.load(f)
    test_preds = list(baseline_preds.values())

with open("results/ehr_test_t5_preds.json", "r") as f:
    t5_preds = json.load(f)
    t5_test_preds = list(t5_preds.values())

with open("EHRSQL_data/mimic_iv/test/label.json", "r") as f:
    test_labels = json.load(f)
    test_labels = list(test_labels.values())

with open("tmp/log_probability_ehr_test_data.pickle", "rb") as f:
    logprobs_test_data = pickle.load(f)


def calc_avg_log_bottom_k(logprobs, t=10):
    """
    Compute the average log probability of the lowest t non-reserved tokens.
    :return: Average of the bottom-t log probabilities
    """
    filtered_logprobs = []
    for item in logprobs:
        token, lp = item.token, item.logprob
        if token.strip().upper() in RESERVED_WORDS:
            continue
        filtered_logprobs.append(lp)

    if not filtered_logprobs:
        return float("-inf")  # Abstain if no meaningful tokens

    # Get bottom-t log probabilities
    bottom_k_logprobs = sorted(filtered_logprobs)[:t]
    return np.mean(bottom_k_logprobs)


def calc_min_logprob(logprobs):
    """
    Compute the average log probability of the lowest t non-reserved tokens.
    :return: Average of the bottom-t log probabilities
    """
    min_token = None
    min_prob = 1000
    for item in logprobs:
        token, lp = item.token, item.logprob
        if token.strip().upper() in RESERVED_WORDS:
            continue
        if lp < min_prob:
            min_prob = lp
            min_token = token
    return (min_token, min_prob)


def should_abstain(logprobs, threshold=-2.5, t=10):
    """
    Determines whether to abstain based on log probability of the bottom-t tokens.
    :return: True if the model should abstain, False otherwise
    """
    avg_logprob = calc_avg_log_bottom_k(logprobs, t)
    return avg_logprob < threshold


def find_empirical_threshold(logprobs_data, abstain_count=425, t=10):
    """
    Determines the empirical threshold `k` based on the bottom-k log probabilities.
    :return: The computed empirical threshold
    """
    bottom_k_probs = []
    for idx, logprob_data in logprobs_data.items():
        bottom_k_probs.append(calc_avg_log_bottom_k(logprob_data, t))

    sorted_probs = sorted(bottom_k_probs)

    # Get the threshold at rank 425
    return (
        sorted_probs[abstain_count - 1]
        if abstain_count <= len(sorted_probs)
        else sorted_probs[-1]
    )


def execute(db_id: str, sql: str, is_gold_sql: bool, timeout: int = 60):
    if is_gold_sql:
        processed_sql = postprocess_gt(sql, db_id=db_id)
    else:
        processed_sql = postprocess_pred(sql, db_id=db_id)

    if processed_sql == "null":
        return "null"
    else:
        try:
            execution_result = func_timeout(
                timeout=timeout,
                func=execute_sql_for_evaluator,
                args=(
                    processed_sql,
                    "EHRSQL_data/mimic_iv/mimic_iv.sqlite",
                ),
            )
            execution_result = process_answer(execution_result, db_id=db_id)
        except FunctionTimedOut:
            execution_result = f"[Error] Timeout in executing SQL: {timeout} sec\nSQL: {processed_sql}\n"
        except Exception as e:
            execution_result = (
                f"[Error] Error in executing SQL: {e}\nSQL: {processed_sql}\n"
            )
        return execution_result


def save_preds(preds, model="baseline", type="threshold"):
    with open(
        f"results/ehr_test_{model}_preds_filtered_{type}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(preds, f, indent=2)


def filter_empirical_threshold(count, is_save=False):
    # Compute threshold from validation set
    empirical_threshold = find_empirical_threshold(
        logprobs_test_data, abstain_count=count, t=10
    )
    print(f"Empirical threshold: {empirical_threshold}")

    filtered_results = []  # List of index of questions that will abstain

    for idx, logprob_data in logprobs_test_data.items():
        if should_abstain(logprob_data, threshold=empirical_threshold):
            filtered_results.append(idx)

    print("Filtered count:", len(filtered_results))

    if is_save:
        for idx, pred_sql in baseline_preds.items():
            if idx in filtered_results:
                baseline_preds[idx] = "null"

        save_preds(baseline_preds, "baseline", "threshold")


def filter_error_query(is_save=False):
    for idx, pred_sql in tqdm(baseline_preds.items()):
        result = execute("mimic_iv", pred_sql, False, timeout=60)
        match = re.search(r"\[Error\]", result)
        if match:
            baseline_preds[idx] = "null"
    if is_save:
        cnt = 0
        for idx, pred_sql in baseline_preds.items():
            if pred_sql == "null":
                cnt += 1
        save_preds(baseline_preds, "baseline", "error")


def is_confident(logprob, thres=0.5):
    return abs(logprob) <= thres


def is_error(sql):
    result = execute("mimic_iv", sql, False, timeout=60)
    match = re.search(r"\[Error\]", result)
    return match


if __name__ == "__main__":
    # list_null = [idx for idx in range(len(test_labels)) if test_labels[idx] == "null"]
    # list_errors = [idx for idx in range(len(test_preds)) if is_error(test_preds[idx])]

    # error_null = set(list_null).intersection(set(list_errors))
    # error_not_null = set(list_errors) - set(list_null)

    # print(len(error_null) / len(list_null) * 100)

    # filter_error_query(is_save=True)
    filter_empirical_threshold(233, is_save=True)
