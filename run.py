import os  # Add this import

from baseline import get_conversation
from const import (
    NULL_QUESTION_DATA_PATH,
    NULL_QUESTION_INDEX_PATH,
    TEST_DATA_PATH,
    TEXT_SQL_DATA_PATH,
    TEXT_SQL_INDEX_PATH,
    null_thres,
)
from model import Model, post_process
from retrieve import Retriever, VectorDB
from utils import generate_classification_answer, get_tokenizer_model, load_data, submit


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
    null_vector_db.initialize()

    text_sql_vector_db = VectorDB(
        dataset_path=TEXT_SQL_DATA_PATH, index_path=TEXT_SQL_INDEX_PATH
    )
    text_sql_vector_db.initialize()

    null_retriever = Retriever(null_vector_db)
    text_sql_retriever = Retriever(text_sql_vector_db)

    myModel = Model()

    # classification model and tokenizer
    model, tokenizer = get_tokenizer_model()
    model.eval()

    final_ret = {}
    for sample in data:
        str_id, question = sample["id"], sample["question"]

        if is_short_question(question):
            final_ret[str_id] = "null"  # Abstain
            continue

        results = null_retriever.retrieve(question)
        if results:
            most_similar_question, distance = results[0]
            if distance <= null_thres:
                final_ret[str_id] = "null"  # Abstain
                continue

        answer = generate_classification_answer(question, model, tokenizer)
        # if the answer is NO then abstain
        if answer == "NO":
            final_ret[str_id] = "null"
        else:
            few_shots = text_sql_retriever.retrieve(question)
            prompt = get_conversation(question, few_shots)
            answer, _ = myModel.ask_chatgpt(prompt)
            final_ret[str_id] = post_process(answer)

    submit(final_ret)
