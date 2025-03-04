import json

from const import RESULT_DIR, TEAM_ID, VALID_DATA_PATH, VALID_LABEL_PATH
from scoring.scorer import Scorer

with open(VALID_DATA_PATH, "r") as f:
    data = json.load(f)

with open(VALID_LABEL_PATH, "r") as f:
    gold_labels = json.load(f)

with open(f"{RESULT_DIR}/team_{TEAM_ID}.json", "r") as f:
    predictions = json.load(f)

scorer = Scorer(
    data=data, predictions=predictions, gold_labels=gold_labels, score_dir="results"
)

print(scorer.get_scores())
