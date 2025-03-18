from baseline import get_conversation
from const import (
    TEST_DATA_PATH,
)
from model import Model, post_process
from retrieve import Retriever
from utils import load_data, submit


def is_short_question(question):
    words = question.split()
    return len(words) <= 2


if __name__ == "__main__":
    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    data = test_data["data"]
    rv = Retriever()
    myModel = Model()
    final_ret = {}
    for sample in data:
        str_id, question = sample["id"], sample["question"]
        if is_short_question(question):
            final_ret[str_id] = "null"  # Abstain
        most_sim_id, most_sim_question, sim_score = rv.retrieve(question)
        if sim_score > 0.75:
            final_ret[str_id] = "null"  # Abstain
        else:
            prompt = get_conversation()
            answer, _ = myModel.ask_chatgpt(prompt)
            final_ret[str_id] = post_process(answer)

    submit(final_ret)
