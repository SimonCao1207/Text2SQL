import os  # Add this import

from baseline import get_conversation
from const import (
    FINETUNED_GPT_MINI,
    NULL_QUESTION_DATA_PATH,
    NULL_QUESTION_INDEX_PATH,
    O3_MINI_GPT,
    TEST_DATA_PATH,
    TEXT_SQL_DATA_PATH,
    TEXT_SQL_INDEX_PATH,
    GPT_4o,
    null_thres,
)
from filter_data import is_error
from model import Model, post_process
from retrieve import Retriever, VectorDB
from utils import generate_classification_answer, get_tokenizer_model, load_data, submit


def is_short_question(question):
    words = question.split()
    return len(words) <= 2


def error_handling(answer, model, prompt, max_attempt=3):
    """Retry generating an answer if an error occurs."""
    if not is_error(answer):
        return answer
    for _ in range(max_attempt):
        answer, _ = model.ask_chatgpt(prompt)
        if not is_error(answer):
            return answer
    return "null"


def initialize_vector_db(dataset_path, index_path, model_name):
    vector_db = VectorDB(
        dataset_path=dataset_path, index_path=index_path, model_name=model_name
    )
    vector_db.initialize()
    return vector_db


if __name__ == "__main__":
    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    data = test_data["data"]

    if not os.path.exists("./data/index"):
        os.makedirs("./data/index")

    null_vector_db = initialize_vector_db(
        dataset_path=NULL_QUESTION_DATA_PATH,
        index_path=NULL_QUESTION_INDEX_PATH,
        model_name="all-MiniLM-L6-v2",
    )
    text_sql_vector_db = initialize_vector_db(
        dataset_path=TEXT_SQL_DATA_PATH,
        index_path=TEXT_SQL_INDEX_PATH,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
    )
    null_retriever = Retriever(null_vector_db)
    text_sql_retriever = Retriever(text_sql_vector_db)

    gpt_mini_model = Model(model=FINETUNED_GPT_MINI)
    gpt_model = Model(model=GPT_4o)
    reasoning_model = Model(model=O3_MINI_GPT)

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
            answer, _ = gpt_model.ask_chatgpt(prompt)

            # Handle errors and post-process the answer
            final_answer = error_handling(answer, reasoning_model, prompt)
            if final_answer != "null":
                final_answer = post_process(final_answer)
            final_ret[str_id] = final_answer

    submit(final_ret)
