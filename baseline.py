import pickle

from const import (
    DB_ID,
    EHR_VALID_DATA_PATH,
    EHR_VALID_LABEL_PATH,
    SYSTEM_PROMPT,
    TABLES_PATH,
)
from model import Model
from utils import create_schema_prompt, get_scores, load_data, load_schema, submit

if __name__ == "__main__":
    # Load data from valid dataset
    valid_data, valid_labels = load_data(EHR_VALID_DATA_PATH, EHR_VALID_LABEL_PATH)

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
    label_y, logprobs = myModel.generate(input_data)

    with open("log_probability_ehr_valid_data.pickle", "wb") as f:
        pickle.dump(logprobs, f, pickle.HIGHEST_PROTOCOL)

    submit(label_y)

    get_scores(valid_data, valid_labels, label_y)
