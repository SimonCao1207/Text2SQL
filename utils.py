import json
import os

import pandas as pd

from const import RESULT_DIR, TEAM_ID
from scoring.scorer import Scorer


def read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file


def write_json(path, file):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, "w+") as f:
        json.dump(file, f)


def load_data(data_path, label_path, is_test=False):
    with open(os.path.join(data_path), "r") as f:
        data = json.load(f)
    if not is_test and label_path is not None:
        with open(os.path.join(label_path), "r") as f:
            labels = json.load(f)
    else:
        labels = {}
    return data, labels


# This function loads and processes a database schema from a JSON file.
def load_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(["column_names", "table_names"], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row["table_names_original"]
        col_names = row["column_names_original"]
        col_types = row["column_types"]
        foreign_keys = row["foreign_keys"]
        primary_keys = row["primary_keys"]
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index > -1:
                schema.append([row["db_id"], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row["db_id"], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append(
                [
                    row["db_id"],
                    tables[first_index],
                    tables[second_index],
                    first_column,
                    second_column,
                ]
            )
    db_schema = pd.DataFrame(
        schema, columns=["Database name", "Table Name", "Field Name", "Type"]
    )
    primary_key = pd.DataFrame(
        p_keys, columns=["Database name", "Table Name", "Primary Key"]
    )
    foreign_key = pd.DataFrame(
        f_keys,
        columns=[
            "Database name",
            "First Table Name",
            "Second Table Name",
            "First Table Foreign Key",
            "Second Table Foreign Key",
        ],
    )
    return db_schema, primary_key, foreign_key


# Generates a string representation of foreign key relationships in a MySQL-like format for a specific database.
def find_foreign_keys_MYSQL_like(foreign, db_id):
    df = foreign[foreign["Database name"] == db_id]
    output = "["
    for index, row in df.iterrows():
        output += (
            row["First Table Name"]
            + "."
            + row["First Table Foreign Key"]
            + " = "
            + row["Second Table Name"]
            + "."
            + row["Second Table Foreign Key"]
            + ", "
        )
    output = output[:-2] + "]"
    if len(output) == 1:
        output = "[]"
    return output


# Creates a string representation of the fields (columns) in each table of a specific database, formatted in a MySQL-like syntax.
def find_fields_MYSQL_like(db_schema, db_id):
    df = db_schema[db_schema["Database name"] == db_id]
    df = df.groupby("Table Name")
    output = ""
    for name, group in df:
        output += "Table " + name + ", columns = ["
        for index, row in group.iterrows():
            output += row["Field Name"] + ", "
        output = output[:-2]
        output += "]\n"
    return output


# Generates a comprehensive textual prompt describing the database schema, including tables, columns, and foreign key relationships.
# Then, add the SQL assumptions for MIMIC-IV
def create_schema_prompt(
    db_id,
    db_schema,
    primary_key,
    foreign_key,
    assumptions,
    is_lower=True,
    add_assumptions=False,
):
    prompt = find_fields_MYSQL_like(db_schema, db_id)
    prompt += "Foreign_keys = " + find_foreign_keys_MYSQL_like(foreign_key, db_id)
    if is_lower:
        prompt = prompt.lower()
    if add_assumptions:
        prompt += "\nSQL Assumptions that you must follow:\n" + assumptions
    return prompt


# Save your api key into json file
def save_api_key(open_ai_key):
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
    api_path = "/tmp/openai_api_key.json"
    json_data = {}
    json_data["key"] = open_ai_key
    with open(api_path, "w") as file:
        json.dump(json_data, file, indent=4)


# Save the filtered predictions to a JSON file
def submit(label, file_name=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    if file_name and isinstance(file_name, str):
        SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, file_name)
    else:
        SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, f"team_{TEAM_ID}.json")

    write_json(SCORING_OUTPUT_DIR, label)


# Get score to evaluate the performance the model
def get_scores(data, gold_labels, predictions):
    scorer = Scorer(
        data=data, predictions=predictions, gold_labels=gold_labels, score_dir="results"
    )
    print()
    print(scorer.get_scores())
