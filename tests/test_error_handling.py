import os
import sys
import unittest

from const import O3_MINI_GPT
from filter_data import is_error
from model import Model, post_process
from run import error_handling, get_conversation

# Add parent directory to path to import filter_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestErrorHandling(unittest.TestCase):
    def test_loop(self):
        reasoning_model = Model(model=O3_MINI_GPT)
        question = "When did the first hospital discharge of patient 10026406 occur?"
        answer = "SELEC * FROM patients"
        answer = post_process(answer)
        prompt = get_conversation(question, few_shots=None)
        final_answer = error_handling(answer, reasoning_model, prompt)
        self.assertFalse(is_error(final_answer))


if __name__ == "__main__":
    unittest.main()
