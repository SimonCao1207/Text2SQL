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

VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, "valid_data.json")
VALID_LABEL_PATH = os.path.join(BASE_DATA_DIR, "valid_label.json")

TRAIN_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "train", "data.json")
TRAIN_LABEL_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "train", "label.json")

# VALID_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "valid", "data.json")
# VALID_LABEL_PATH = os.path.join(EHRSQL_DATA_DIR, DB_ID, "valid", "label.json")

# PROMPT
SYSTEM_PROMPT = "Given the following SQL tables and SQL assumptions you must follow, your job is to write queries given a user’s request.\n IMPORTANT: If you think you cannot predict the SQL accurately, you must answer with 'null'."
