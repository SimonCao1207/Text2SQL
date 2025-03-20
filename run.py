import os  # Add this import

from baseline import get_conversation
from const import (
    NULL_QUESTION_DATA_PATH,
    NULL_QUESTION_INDEX_PATH,
    TEST_DATA_PATH,
)
from model import Model, post_process
from retrieve import Retriever, VectorDB
from utils import load_data, submit


def is_short_question(question):
    words = question.split()
    return len(words) <= 2


if __name__ == "__main__":
    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    data = test_data["data"]

    if not os.path.exists("./data/index"):
        os.makedirs("./data/index")

    null_vector_db = VectorDB(
        dataset_path=NULL_QUESTION_DATA_PATH, index_path=NULL_QUESTION_INDEX_PATH
    )

    if not null_vector_db.index_path.exists():
        null_vector_db.build_index()
    else:
        null_vector_db.load_index()

    null_retriever = Retriever(null_vector_db)
    thres = 0.4

    myModel = Model()
    final_ret = {}
    for sample in data:
        str_id, question = sample["id"], sample["question"]
        if is_short_question(question):
            final_ret[str_id] = "null"  # Abstain
        results = null_retriever.retrieve(question)
        if results:
            most_similar_question, distance = results[0]
            if distance <= thres:
                final_ret[str_id] = "null"  # Abstain
        else:
            prompt = get_conversation(question)
            answer, _ = myModel.ask_chatgpt(prompt)
            final_ret[str_id] = post_process(answer)

    submit(final_ret)
