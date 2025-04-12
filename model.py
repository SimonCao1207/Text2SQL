import json
import os
import re

import openai
from dotenv import load_dotenv
from openai import OpenAI
from opik import track
from tqdm import tqdm

from const import FINETUNED_GPT_MINI
from utils import save_api_key

load_dotenv()
open_ai_key = os.environ.get("OPENAI_API_KEY")
if open_ai_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
save_api_key(open_ai_key)
client = OpenAI(api_key=open_ai_key)


def post_process(answer):
    answer = answer.replace("\n", " ")
    answer = re.sub("[ ]+", " ", answer)
    answer = answer.replace("```sql", "").replace("```", "").strip()
    return answer


class Model:
    def __init__(self, model=FINETUNED_GPT_MINI):
        self.model = model
        current_real_dir = os.getcwd()
        # current_real_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_real_dir, "/tmp/openai_api_key.json")

        if os.path.isfile(target_dir):
            with open(target_dir, "rb") as f:
                openai.api_key = json.load(f)["key"]
        if not os.path.isfile(target_dir) or openai.api_key == "":
            raise Exception("Error: no API key file found.")

    @track
    def ask_chatgpt(
        self,
        prompt,
    ):
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt,
        )
        return response.choices[0].message.content

    def generate(self, input_data):
        """
        Arguments:
            input_data: list of python dictionaries containing 'id' and 'input'
        Returns:
            labels: python dictionary containing sql prediction or 'null' values associated with ids
        """

        labels = {}

        for sample in tqdm(input_data):
            answer = self.ask_chatgpt(sample["input"])
            labels[sample["id"]] = post_process(answer[0])

        return labels
