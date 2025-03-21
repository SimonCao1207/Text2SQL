import json

import torch
from peft.peft_model import PeftModel
from tqdm import tqdm

from abstain_finetune import get_tokenizer_model
from const import (
    DB_ID,
    EHR_TEST_DATA_PATH,
    EHR_TEST_LABEL_PATH,
    PROMPT_CLASSIFICATION,
    TABLES_PATH,
)
from utils import create_schema_prompt, load_data, load_schema


def main(is_save="False"):
    model, tokenizer = get_tokenizer_model("defog/sqlcoder-7b-2", None, None)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    peft_model = PeftModel.from_pretrained(
        model, "outputs/second_run/checkpoint-6400"
    ).to("cuda")
    peft_model.eval()

    test_data, test_label = load_data(EHR_TEST_DATA_PATH, EHR_TEST_LABEL_PATH)
    test_data = test_data["data"]

    with open(PROMPT_CLASSIFICATION, "r") as f:
        prompt = f.read()

    db_schema, primary_key, foreign_key = load_schema(TABLES_PATH)
    table_prompt = create_schema_prompt(DB_ID, db_schema, primary_key, foreign_key, "")

    total = 0
    n_yes = 0
    n_no = 0

    no_as_yes = 0
    no_as_no = 0
    yes_as_no = 0
    yes_as_yes = 0
    n_invalid = 0
    for i, data in enumerate(tqdm(test_data)):
        id_ = data["id"]
        question = data["question"]
        question = prompt.format(
            user_question=question, table_metadata_string=table_prompt
        )

        total += 1
        if test_label[id_] == "null":
            n_no += 1
            answer = "NO"
            data["answer"] = answer
        else:
            n_yes += 1
            answer = "YES"
            data["answer"] = answer

        question_tokenized = tokenizer.encode(question, add_special_tokens=True)
        question_tokenized += tokenizer.encode("### Answer" + "\n")
        question_tokenized = torch.Tensor(question_tokenized).long().view(1, -1)
        question_tokenized = question_tokenized.to("cuda")
        attention_mask = torch.ones_like(question_tokenized).to("cuda")

        prompt_len = question_tokenized.shape[1]
        with torch.no_grad():
            outputs = peft_model.generate(
                input_ids=question_tokenized,
                attention_mask=attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            data["generated"] = text

            if text == "YES" and answer == "YES":
                yes_as_yes += 1
            elif text == "NO" and answer == "NO":
                no_as_no += 1
            elif text == "NO" and answer == "YES":
                yes_as_no += 1
            elif text == "YES" and answer == "NO":
                no_as_yes += 1
            else:
                n_invalid += 1
                print(f"Generated answer: {text}")

    correct = no_as_no + yes_as_yes
    print(f"Total number of data: {total}")
    print(f"Number of YES: {n_yes}")
    print(f"Number of NO: {n_no}")

    print(
        f"Number of correct predictions and accuracy: {correct, 100 * correct / total}"
    )

    print(f"Number of NO as YES: {no_as_yes}")
    print(f"Number of YES as NO: {yes_as_no}")
    print(f"Number of YES as YES: {yes_as_yes}")
    print(f"Number of NO as NO: {no_as_no}")
    print(f"Number of INVALID predictions: {n_invalid}")

    if is_save:
        output_path = "classification_results.json"
        with open(output_path, "w") as f:
            json.dump(test_data, f)


if __name__ == "__main__":
    main()
