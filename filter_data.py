import json
import re

from func_timeout import FunctionTimedOut, func_timeout

from scoring.utils import (
    execute_sql_for_evaluator,
    postprocess_gt,
    postprocess_pred,
    process_answer,
)

model = "baseline"
with open(f"results/ehr_test_{model}_preds.json", "r") as f:
    preds = json.load(f)


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


for idx, pred_sql in preds.items():
    result = execute("mimic_iv", pred_sql, False, timeout=60)
    match = re.search(r"\[Error\]", result)
    if match:
        preds[idx] = "null"

cnt = 0
for idx, pred_sql in preds.items():
    if pred_sql == "null":
        cnt += 1
with open(f"results/ehr_test_{model}_preds_filtered.json", "w", encoding="utf-8") as f:
    json.dump(preds, f, indent=2)
