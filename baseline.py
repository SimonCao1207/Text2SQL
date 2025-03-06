import json
import os

from const import (
    DB_ID,
    SYSTEM_PROMPT,
    TABLES_PATH,
    VALID_DATA_PATH,
    VALID_LABEL_PATH,
)
from model import Model
from utils import create_schema_prompt, get_scores, load_schema, submit

if __name__ == "__main__":
    # Load data
    with open(os.path.join(VALID_DATA_PATH), "r") as f:
        valid_data = json.load(f)

    with open(os.path.join(VALID_LABEL_PATH), "r") as f:
        valid_labels = json.load(f)

    print("Valid data:", (len(valid_data["data"]), len(valid_labels)))
    print(valid_data.keys())
    print(valid_labels[list(valid_labels.keys())[0]])

    # Load SQL assumptions for MIMIC-IV
    assumptions = open("database/mimic_iv_assumption.txt", "r").read()

    db_schema, primary_key, foreign_key = load_schema(TABLES_PATH)
    table_prompt = create_schema_prompt(
        DB_ID, db_schema, primary_key, foreign_key, assumptions
    )
    myModel = Model()
    data = valid_data["data"]

    input_data = []
    for sample in data:
        sample_dict = {}
        sample_dict["id"] = sample["id"]
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + table_prompt}
        ]

        def user_question_wrapper(question):
            return "\n\n" + f"""NLQ: \"{question}\"\nSQL: """

        conversation.append(
            {"role": "user", "content": user_question_wrapper(sample["question"])}
        )
        sample_dict["input"] = conversation
        input_data.append(sample_dict)

    # Generate answer(SQL) from chatGPT
    label_y = myModel.generate(input_data)

    submit(label_y)

    get_scores(valid_data, valid_labels, label_y)
