import json
import re

from func_timeout import FunctionTimedOut, func_timeout

from scoring.utils import (
    execute_sql_for_evaluator,
    postprocess_gt,
    postprocess_pred,
    process_answer,
)

with open("EHRSQL_data/mimic_iv/test/label.json", "r") as f:
    test_labels = json.load(f)
    test_labels = list(test_labels.values())


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


def is_error(sql):
    result = execute("mimic_iv", sql, False, timeout=60)
    match = re.search(r"\[Error\]", result)
    return match


def is_empty(sql):
    result = execute("mimic_iv", sql, False, timeout=60)
    return result == "[]" or result == ""
