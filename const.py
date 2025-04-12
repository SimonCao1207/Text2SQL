import os

# Team ID
TEAM_ID = "12"

# Directory paths for database, results and scoring program
DB_ID = "mimic_iv"
BASE_DATA_DIR = "data"
EHRSQL_DATA_DIR = "EHRSQL_data"
RESULT_DIR = "results"
SCORING_DIR = "scoring"

TABLES_PATH = os.path.join("database", "tables.json")
DB_PATH = os.path.join("database", DB_ID, f"{DB_ID}.sqlite")  # Database path

# Original data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, "valid_data.json")
VALID_LABEL_PATH = os.path.join(BASE_DATA_DIR, "valid_label.json")
TEST_DATA_PATH = os.path.join(BASE_DATA_DIR, "test_data.json")

# EHRSQL data
EHR_TRAIN_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "train", "data.json")
EHR_TRAIN_LABEL_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "train", "label.json")

EHR_VALID_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "valid", "data.json")
EHR_VALID_LABEL_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "valid", "label.json")

EHR_TEST_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "test", "data.json")
EHR_TEST_LABEL_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "test", "label.json")

# Null question data and index path
NULL_QUESTION_DATA_PATH = os.path.join(BASE_DATA_DIR, "ehr_null_data.csv")
NULL_QUESTION_INDEX_PATH = os.path.join(BASE_DATA_DIR, "index/ehr_null.index")

# Text SQL data and index path
TEXT_SQL_DATA_PATH = os.path.join(BASE_DATA_DIR, "text_sql.csv")
TEXT_SQL_INDEX_PATH = os.path.join(BASE_DATA_DIR, "index/text_sql.index")

# Threshold constant
null_thres = 0.4

# PROMPT
SYSTEM_PROMPT = "Given the following SQL tables and SQL assumptions you must follow, your job is to write queries given a user's request.\n You will first be shown examples of natural language questions (NLQs) alongside their corresponding SQL queries. At the end, you will be presented with one final question. Generate the SQL query for this final question only. IMPORTANT: If you think you cannot predict the SQL accurately, you must answer with 'null'."
PROMPT_CLASSIFICATION = "prompt_classification.md"

# Model paths
PRETRAINED_MODEL_PATH = "defog/sqlcoder-7b-2"
LORA_PATH = "outputs/merged_run/checkpoint-6400"

# Model name
FINETUNED_GPT_MINI = "ft:gpt-4o-mini-2024-07-18:personal::B7xHlv2W"
O3_MINI_GPT = "o3-mini"
GPT_4o = "gpt-4o"
