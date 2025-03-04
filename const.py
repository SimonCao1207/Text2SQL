import os

# Student ID
TEAM_ID = "12"

# Directory paths for database, results and scoring program
DB_ID = "mimic_iv"
BASE_DATA_DIR = "data"
RESULT_DIR = "results"
SCORING_DIR = "scoring"

# File paths for the dataset and labels
TABLES_PATH = os.path.join("database", "tables.json")  # JSON containing database schema
VALID_DATA_PATH = os.path.join(
    BASE_DATA_DIR, "valid_data.json"
)  # JSON file for validation data
VALID_LABEL_PATH = os.path.join(
    BASE_DATA_DIR, "valid_label.json"
)  # JSON file for validation labels (for evaluation)
DB_PATH = os.path.join("data", DB_ID, f"{DB_ID}.sqlite")  # Database path
