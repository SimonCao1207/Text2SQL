import pickle

from const import (
    DB_ID,
    EHR_TEST_DATA_PATH,
    EHR_TEST_LABEL_PATH,
    SYSTEM_PROMPT,
    TABLES_PATH,
    text_sql_thres,
)
from model import Model
from utils import create_schema_prompt, get_scores, load_data, load_schema, submit


def add_few_shots(prompt, few_shots):
    most_similar_item, distance = few_shots[0]
    question, sql = most_similar_item["question"], most_similar_item["sql"]
    if distance <= text_sql_thres:
        prompt += f"\n\nExample of your response:\nNLQ: {question}\nSQL: {sql}"
    return prompt


def get_conversation(question, few_shots=None):
    # Load SQL assumptions for MIMIC-IV
    assumptions = open("database/mimic_iv_assumption.txt", "r").read()

    db_schema, primary_key, foreign_key = load_schema(TABLES_PATH)
    table_prompt = create_schema_prompt(
        DB_ID, db_schema, primary_key, foreign_key, assumptions
    )
    if few_shots:
        table_prompt += add_few_shots(prompt=table_prompt, few_shots=few_shots)

    # print(table_prompt)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + table_prompt}
    ]

    def user_question_wrapper(question):
        return "\n\n" + f"""NLQ: \"{question}\"\nSQL: """

    conversation.append({"role": "user", "content": user_question_wrapper(question)})
    return conversation


def run_baseline():
    # Load data from ehr test dataset
    test_data, test_labels = load_data(EHR_TEST_DATA_PATH, EHR_TEST_LABEL_PATH)

    myModel = Model()
    data = test_data["data"]

    input_data = []
    for sample in data:
        sample_dict = {}
        sample_dict["id"] = sample["id"]
        sample_dict["input"] = get_conversation(sample["question"], few_shots=None)
        input_data.append(sample_dict)

    # Generate answer(SQL) from chatGPT
    label_y, logprobs = myModel.generate(input_data)
    return test_data, test_labels, label_y, logprobs


if __name__ == "__main__":
    test_data, test_labels, label_y, logprobs = run_baseline()

    # TODO: use logprobs logprobs to estimate confidence and decide whether to abstain.

    with open("log_probability_ehr_test_data.pickle", "wb") as f:
        pickle.dump(logprobs, f, pickle.HIGHEST_PROTOCOL)

    submit(label_y)

    get_scores(test_data, test_labels, label_y)
