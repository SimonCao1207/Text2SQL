import json
import pickle
import re

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout

from const import RESERVED_WORDS
from scoring.utils import (
    execute_sql_for_evaluator,
    postprocess_gt,
    postprocess_pred,
    process_answer,
)

model = "baseline"
with open(f"results/ehr_test_{model}_preds.json", "r") as f:
    preds = json.load(f)

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
        if token.upper() in RESERVED_WORDS:
            continue
        filtered_logprobs.append(lp)

    if not filtered_logprobs:
        return float("-inf")  # Abstain if no meaningful tokens

    # Get bottom-t log probabilities
    bottom_k_logprobs = sorted(filtered_logprobs)[:t]
    return np.mean(bottom_k_logprobs)


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


def save_preds(preds, type="threshold"):
    with open(
        f"results/ehr_test_{model}_preds_filtered_{type}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(preds, f, indent=2)


def filter_empirical_threshold(count):
    # Compute threshold from validation set
    empirical_threshold = find_empirical_threshold(
        logprobs_test_data, abstain_count=count, t=10
    )
    print(f"Empirical threshold: {empirical_threshold}")

    filtered_results = []  # List of index of questions that will abstain

    for idx, logprob_data in logprobs_test_data.items():
        if should_abstain(logprob_data):
            filtered_results.append(idx)

    for idx, pred_sql in preds.items():
        if idx in filtered_results:
            preds[idx] = "null"

    print("Filtered count:", len(filtered_results))

    save_preds(preds, "threshold")


def filter_error_query():
    for idx, pred_sql in preds.items():
        result = execute("mimic_iv", pred_sql, False, timeout=60)
        match = re.search(r"\[Error\]", result)
        if match:
            preds[idx] = "null"

    cnt = 0
    for idx, pred_sql in preds.items():
        if pred_sql == "null":
            cnt += 1
    save_preds(preds, "error")


if __name__ == "__main__":
    abstain_count = 425
    filter_empirical_threshold(abstain_count)
