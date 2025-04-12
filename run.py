import json
import logging
import os  # Add this import
import time

from tqdm import tqdm

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
from filter_data import is_empty, is_error
from model import Model, post_process
from retrieve import Retriever, VectorDB
from utils import generate_classification_answer, get_tokenizer_model, load_data, submit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("process.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def save_checkpoint(data, iteration):
    """Save checkpoint of current progress."""
    checkpoint_dir = "./results"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{checkpoint_dir}/checkpoint_{iteration}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(data, f)

    logger.info(f"Checkpoint saved: {filename}")
    return filename


def is_short_question(question):
    words = question.split()
    return len(words) <= 2


def error_handling(answer, model, prompt, max_attempt=3):
    """Retry generating an answer if an error occurs."""
    if not is_error(answer):
        return answer
    for _ in range(max_attempt):
        answer = model.ask_chatgpt(prompt)
        if not is_error(answer):
            logger.info("\t[Error_handling] Success!")
            return answer
        logger.info("\t[Error_handling] Failed!")
    return "null"


def initialize_vector_db(dataset_path, index_path, model_name):
    vector_db = VectorDB(
        dataset_path=dataset_path, index_path=index_path, model_name=model_name
    )
    vector_db.initialize()
    return vector_db


if __name__ == "__main__":
    logger.info("Starting the evaluation process...")
    test_data, _ = load_data(TEST_DATA_PATH, None, is_test=True)
    data = test_data["data"]
    logger.info(f"Loaded {len(data)} test samples")

    if not os.path.exists("./data/index"):
        os.makedirs("./data/index")

    logger.info("Initializing vector databases...")
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
    logger.info("Vector databases initialized successfully")

    logger.info("Loading models...")
    gpt_mini_model = Model(model=FINETUNED_GPT_MINI)
    gpt_model = Model(model=GPT_4o)
    reasoning_model = Model(model=O3_MINI_GPT)

    # classification model and tokenizer
    model, tokenizer = get_tokenizer_model()
    model.eval()
    logger.info("Models loaded successfully")

    final_ret = {}
    checkpoint_interval = max(1, len(data) // 10)  # Save 10 checkpoints

    num_short_questions = 0
    num_abstain_by_cls = 0
    num_abstain_by_null_index = 0
    num_abstain_by_empty_answer = 0
    num_error = 0
    num_abstain_by_error_handling = 0
    num_success = 0

    for i, sample in enumerate(tqdm(data, desc="Processing samples", total=len(data))):
        str_id, question = sample["id"], sample["question"]
        logger.info(f"Processing sample {i + 1}/{len(data)}, ID: {str_id}")

        if is_short_question(question):
            final_ret[str_id] = "null"  # Abstain
            logger.info(f"Sample {str_id}: Short question detected, abstaining")
            num_short_questions += 1
            continue

        results = null_retriever.retrieve(question)
        if results:
            most_similar_question, distance = results[0]
            logger.info(
                f"Sample {str_id}: Similar null question found, distance: {distance}"
            )
            if distance <= null_thres:
                final_ret[str_id] = "null"  # Abstain
                logger.info(f"Sample {str_id}: Distance below threshold, abstaining")
                num_abstain_by_null_index += 1
                continue
        answer = generate_classification_answer(question, model, tokenizer)
        # if the answer is NO then abstain
        if answer == "NO":
            final_ret[str_id] = "null"
            logger.info(f"Sample {str_id}: Classification result is NO, abstaining")
            num_abstain_by_cls += 1
        else:
            logger.info(f"Sample {str_id}: Getting few-shot examples")
            prompt = get_conversation(question, few_shots=None)
            few_shots = text_sql_retriever.retrieve(question)
            prompt = get_conversation(question, few_shots)

            logger.info(f"Sample {str_id}: Generating answer with GPT-4o model")
            answer = gpt_model.ask_chatgpt(prompt)
            is_error_flag = is_error(answer)

            # Handle errors and post-process the answer
            final_answer = post_process(answer)
            if is_empty(final_answer):
                final_ret[str_id] = "null"
                logger.info(
                    f"Sample {str_id}: Empty answer after post-processing, abstaining"
                )
                num_abstain_by_empty_answer += 1
                continue

            logger.info(f"Sample {str_id}: Handling potential errors")
            if is_error_flag:
                final_answer = error_handling(final_answer, reasoning_model, prompt)

            updated_error_flag = is_error(final_answer)
            if updated_error_flag and final_answer == "null":
                num_abstain_by_error_handling += 1
            elif updated_error_flag and final_answer != "null":
                num_error += 1  # num_error should be 0!
            else:
                num_success += 1
            final_ret[str_id] = final_answer

        # Save checkpoint at regular intervals
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_file = save_checkpoint(final_ret, i + 1)

    # Log final statistics
    logger.info("=== Statistics Summary ===")
    logger.info(f"Total samples processed: {len(data)}")
    logger.info(f"Total Success: {num_success}")
    logger.info(f"Short questions (abstained): {num_short_questions}")
    logger.info(f"Abstained by classification: {num_abstain_by_cls}")
    logger.info(f"Abstained by null index similarity: {num_abstain_by_null_index}")
    logger.info(f"Abstained due to empty answers: {num_abstain_by_empty_answer}")
    logger.info(f"Errors detected: {num_error}")  # This valud should be 0!
    logger.info(f"Errors handled by fallback: {num_abstain_by_error_handling}")
    logger.info(
        f"Total abstentions: {num_short_questions + num_abstain_by_cls + num_abstain_by_null_index + num_abstain_by_empty_answer + num_abstain_by_error_handling}"
    )

    logger.info("Processing complete, submitting final results")
    submit(final_ret)
    logger.info("Final results submitted")
