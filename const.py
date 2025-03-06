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

VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, "valid_data.json")
VALID_LABEL_PATH = os.path.join(BASE_DATA_DIR, "valid_label.json")

# TRAIN_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, "train", "data.json")
# TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, "label", "label.json")

# VALID_DATA_PATH = os.path.join(EHRSQL_DATA_DIR, "valid", "data.json")
# VALID_LABEL_PATH = os.path.join(BASE_DATA_DIR, "label", "label.json")

DB_PATH = os.path.join("database", DB_ID, f"{DB_ID}.sqlite")  # Database path
