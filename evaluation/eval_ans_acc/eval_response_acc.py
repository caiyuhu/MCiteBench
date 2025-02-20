import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import openai
from openai import OpenAI
from pydantic import BaseModel
from requests.exceptions import RequestException
from tqdm import tqdm

from utils.utils import load_config, load_jsonl

MODEL_NAME = "gpt-4o-2024-11-20"

INPUT_DIR = "../../output/responses"
OUTPUT_DIR = "../../output/answer_acc"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2
MAX_WORKERS = 8

config = load_config("../../configs/config.json")
OPENAI_API_KEY = config["OPENAI_API_KEY"]

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# ------------------------------
# Setup Logging
# ------------------------------
logging.basicConfig(
    filename="eval_response_acc_all.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ------------------------------
# Data Model for parse
# ------------------------------
class Rating(BaseModel):
    answer_rating: int

# ------------------------------
# explanation prompt
# ------------------------------
explanation_sys_prompt_path = "../../prompts/eval_explanation_acc.txt"


with open(explanation_sys_prompt_path, "r", encoding="utf-8") as file:
    explanation_sys_prompt = file.read()

def build_explanation_prompt(
    question: str,
    correct_answer: str,
    response: str,
) -> str:
    prompt = f"""
Question: {question}
[Reference Answer Start]
{correct_answer}
[Reference Answer End]
[Assistant’s Answer Start]
{response}
[Assistant’s Answer End]
"""
    return prompt


# ------------------------------
# locating prompt
# ------------------------------
locating_sys_prompt_path = "../../prompts/eval_locating_acc.txt"

with open(locating_sys_prompt_path, "r", encoding="utf-8") as file:
    locating_sys_prompt = file.read()


def build_locating_prompt(
    question: str,
    explanation: str,
    correct_answer: str,
    response: str,
) -> str:
    return (
        f"Question: {question}\n"
        f"Reference answer: {correct_answer}\n"
        f"Explanation: {explanation}\n"
        f"Assistant’s answer: {response}\n"
        "Answer:\n"
    )

# ------------------------------
# 核心评分函数: 根据 question_type 判断两套逻辑
# ------------------------------
def evaluate_response(question: Dict, client: OpenAI) -> Dict:
    if "answer_rating" in question:
        return question
    
    q_type = question.get("question_type", "").strip().lower()

    # ============== locating question ==============
    if q_type == "locating":
        q_str = question.get("question", "").strip()
        explanation = question.get("meta_data", {}).get("explanation", "").strip()

        correct_answer = question.get("answer", "").strip()
        response_text = question.get("response", "").strip()

        if (
            not q_str
            or not explanation
            or not correct_answer
            or not response_text
        ):
            logging.warning(
                f"[locating] Missing fields in entry {question.get('question_id','N/A')}. Skipping."
            )
            question["answer_rating"] = -1
            return question

        locating_prompt = build_locating_prompt(
            question=q_str,
            explanation=explanation,
            correct_answer=correct_answer,
            response=response_text
        )


        retries = 0
        while retries < MAX_RETRIES:
            try:
                completion = client.beta.chat.completions.parse(
                    model=MODEL_NAME, 
                    messages=[
                        {"role": "system", "content": locating_sys_prompt},
                        {"role": "user", "content": locating_prompt},
                    ],
                    response_format=Rating,  
                    temperature=0,
                )

                if (
                    hasattr(completion.choices[0].message, "refusal")
                    and completion.choices[0].message.refusal
                ):
                    logging.info(f"[locating] Refusal for question {question.get('question_id')}.")
                    question["answer_rating"] = -1
                else:
                    parsed = completion.choices[0].message.parsed
                    question["answer_rating"] = parsed.answer_rating
                return question

            except (
                openai.error.RateLimitError,
                openai.error.APIError,
                openai.error.Timeout,
                RequestException,
            ) as e:
                logging.error(f"[locating] API error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
                retries += 1
                time.sleep(BACKOFF_FACTOR ** retries)
            except Exception as e:
                logging.error(f"[locating] Unexpected error during evaluation: {e}")
                question["answer_rating"] = -1
                return question

        logging.error(f"[locating] Failed to get rating after {MAX_RETRIES} retries.")
        question["answer_rating"] = -1
        return question

    # ============== explanation question ==============
    elif q_type == "explanation":
        q_str = question.get("question", "").strip()
        correct_answer = question.get("answer", "").strip()
        response_text = question.get("response", "").strip()

        if (
            not q_str
            or not correct_answer
            or not response_text
        ):
            logging.warning(
                f"[explanation] Missing fields in entry {question.get('question_id','N/A')}. Skipping."
            )
            question["answer_rating"] = -1
            return question

        explanation_prompt = build_explanation_prompt(
            question=q_str,
            correct_answer=correct_answer,
            response=response_text
        )

        retries = 0
        while retries < MAX_RETRIES:
            try:
                completion = client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": explanation_sys_prompt},
                        {"role": "user", "content": explanation_prompt},
                    ],
                    response_format=Rating,
                    temperature=0,
                )

                if (
                    hasattr(completion.choices[0].message, "refusal")
                    and completion.choices[0].message.refusal
                ):
                    logging.info(f"[explanation] Refusal for question {question.get('question_id')}.")
                    question["answer_rating"] = -1
                else:
                    parsed = completion.choices[0].message.parsed
                    question["answer_rating"] = parsed.answer_rating

                return question

            except (
                openai.error.RateLimitError,
                openai.error.APIError,
                openai.error.Timeout,
                RequestException,
            ) as e:
                logging.error(f"[explanation] API error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
                retries += 1
                time.sleep(BACKOFF_FACTOR ** retries)
            except Exception as e:
                logging.error(f"[explanation] Unexpected error during evaluation: {e}")
                question["answer_rating"] = -1
                return question

        logging.error(f"[explanation] Failed to get rating after {MAX_RETRIES} retries.")
        question["answer_rating"] = -1
        return question

    else:
        logging.warning(
            f"Unknown or missing question_type in entry {question.get('question_id','N/A')}. Skipping."
        )
        question["answer_rating"] = -1
        return question

# ------------------------------
# JSONL file processing
# ------------------------------
def process_jsonl_file(input_file: str, output_file: str, client: OpenAI):
    """
    Read a JSONL file, check for already processed question_id, process unprocessed entries in parallel using multiple threads, and write back to the output file.
    """
    data = load_jsonl(input_file)
    original_count = len(data)
    if os.path.exists(output_file):
        existing_data = load_jsonl(output_file)
        existing_ids = {
            entry.get("question_id")
            for entry in existing_data
            if "question_id" in entry
        }
        data_to_process = [q for q in data if q.get("question_id") not in existing_ids]
        duplicates_skipped = original_count - len(data_to_process)
        if duplicates_skipped > 0:
            logging.info(
                f"Skipped {duplicates_skipped} already processed entries from {input_file} based on 'question_id'."
            )
            print(
                f"Skipped {duplicates_skipped} already processed entries from {Path(input_file).name} based on 'question_id'."
            )
    else:
        data_to_process = data
        existing_data = []
        existing_ids = set()

    processed_count = 0
    skipped_entries = 0

    if data_to_process:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(
            output_file, "a", encoding="utf-8"
        ) as f:
            future_to_item = {
                executor.submit(evaluate_response, q, client): q
                for q in data_to_process
            }

            for future in tqdm(
                as_completed(future_to_item),
                total=len(future_to_item),
                desc=f"Processing {Path(input_file).name}",
            ):
                item = future_to_item[future]
                try:
                    ret = future.result()
                    f.write(json.dumps(ret, ensure_ascii=False) + "\n")
                    f.flush()
                    processed_count += 1
                except Exception as e:
                    logging.error(
                        f"Error processing entry {item.get('question_id','N/A')}: {e}"
                    )
                    item["answer_rating"] = -1
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    skipped_entries += 1
    else:
        logging.info(f"No new entries to process in {input_file}.")
        print(f"No new entries to process in {Path(input_file).name}.")

    logging.info(
        f"Finished processing {Path(input_file).name}: {processed_count} entries processed, {skipped_entries} entries skipped."
    )
    print(
        f"Finished processing {Path(input_file).name}: {processed_count} entries processed, {skipped_entries} entries skipped."
    )

    logging.info(f"Saved processed data to {output_file}.")
    print(f"Saved processed data to {output_file}.")


def process_all_jsonl_in_directory(input_dir: str, output_dir: str, client: OpenAI):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        logging.warning(f"No JSONL found in {input_dir}")
        print(f"No JSONL found in {input_dir}")
        return

    for jfile in jsonl_files:
        out_file = output_path / jfile.name
        print(f"Processing: {jfile} -> {out_file}")
        logging.info(f"Processing {jfile} -> {out_file}")
        process_jsonl_file(str(jfile), str(out_file), client)

# ------------------------------
# main
# ------------------------------
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    process_all_jsonl_in_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        client=client
    )

if __name__ == "__main__":
    main()