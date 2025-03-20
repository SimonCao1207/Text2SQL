import json
import os
import re

import openai
from dotenv import load_dotenv
from openai import OpenAI
from opik import track
from tqdm import tqdm

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
    def __init__(self):
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
        model="ft:gpt-4o-mini-2024-07-18:personal::B7xHlv2W",
        temperature=0.6,
    ):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=prompt,
            logprobs=True,
            top_logprobs=5,
        )
        return (
            response.choices[0].message.content,
            response.choices[0].logprobs.content,  # type: ignore
        )

    def generate(self, input_data):
        """
        Arguments:
            input_data: list of python dictionaries containing 'id' and 'input'
        Returns:
            labels: python dictionary containing sql prediction or 'null' values associated with ids
            logprobs : python dictionary containing logprobs associated with ids
        """

        labels = {}
        logprobs = {}

        for sample in tqdm(input_data):
            answer = self.ask_chatgpt(sample["input"])
            labels[sample["id"]] = post_process(answer[0])
            logprobs[sample["id"]] = answer[1]

        return labels, logprobs
