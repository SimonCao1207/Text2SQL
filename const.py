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
text_sql_thres = 0.2

# PROMPT
SYSTEM_PROMPT = "Given the following SQL tables and SQL assumptions you must follow, your job is to write queries given a user’s request.\n IMPORTANT: If you think you cannot predict the SQL accurately, you must answer with 'null'."
PROMPT_CLASSIFICATION = "prompt_classification.md"

RESERVED_WORDS = [
    "SELECT",
    "AS",
    "IN",
    "COUNT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "DROP",
    "ALTER",
    "JOIN",
    "ON",
    "GROUP",
    "ORDER",
    "HAVING",
    "LIMIT",
    "UNION",
    "DISTINCT",
    "INDEX",
    "TABLE",
    "VIEW",
    "TRIGGER",
    "PRIMARY KEY",
    "FOREIGN KEY",
    "NULL",
    "NOT NULL",
    "UNIQUE",
    "CHECK",
    "DEFAULT",
    "INDEX",
    "SEQUENCE",
    "EXEC",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "CAST",
    "CHAR",
    "VARCHAR",
    "BOOLEAN",
    "INTEGER",
    "DATE",
    "INTERVAL",
    "TIME",
    "TIMESTAMP",
    "YEAR",
    "MONTH",
    "DAY",
    "HOUR",
    "MINUTE",
    "SECOND",
    "ZONE",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "TRUE",
    "FALSE",
]
