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

# PROMPT
SYSTEM_PROMPT = "Given the following SQL tables and SQL assumptions you must follow, your job is to write queries given a user’s request.\n IMPORTANT: If you think you cannot predict the SQL accurately, you must answer with 'null'."
