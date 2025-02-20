import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from requests.exceptions import RequestException
from tqdm import tqdm

from utils.utils import load_config, load_jsonl


MODEL_NAME = "gpt-4o-2024-11-20"

INPUT_DIR = "../../output/responses"
OUTPUT_DIR = "../../output/extracted_sen_cit_list"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2
MAX_WORKERS = 8

OPENAI_API_KEY = load_config("../../configs/config.json")["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# ------------------------------
# Setup Logging
# ------------------------------
logging.basicConfig(
    filename="extract_sen_cit_list.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# ------------------------------
# Data Models for parse
# ------------------------------
class SentenceCitationPair(BaseModel):
    Sentence: str
    Citation: List[str]


class SentenceCitationList(BaseModel):
    SentenceCitationList: List[SentenceCitationPair]


# ------------------------------
# Helper Functions
# ------------------------------

def build_prompt(response: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", response)

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    formatted_response = "\n".join(
        [f"{sentence}" for i, sentence in enumerate(sentences)]
    )

    prompt = (
        "You are an impartial judge tasked with extracting sentences and their corresponding citations from AI-generated responses. Analyze the following response and split it into multiple sentence-citation pairs. For each pair, extract the original sentence and the citations associated with it."
        "- Citations can be in the form of [1], [2], [3], or textual references like Figure 3, Table 3, etc. Ensure that citations are correctly identified and formatted.\n"
        "- If a citation refers to a subfigure or subtable (e.g., Figure 3(a), Table 3(a)), only extract the main figure or table number (e.g., Figure 3, Table 3).\n"
        "Extract the original sentence exactly as it appears in the response, do not modify any content or rephrase the sentence."
        "- For each sentence, extract only one sentence-citation pair, even if there are multiple citations in the same sentence. Do not split a sentence into multiple pairs.\n"
        "- If a sentence does not contain any citations, do not include it in the output.\n"
        "- Each line in the Model Response is a separate sentence.\n"
        "- If the model's response contains citations Reference the model defined itself, extract the corresponding sentence and citation.\n"
        "- If the model's response contains repetition (e.g., repeated sentences or an unending loop), only include the valid, non-repetitive sentences and their corresponding citations. Ignore any repetitive or nonsensical parts."
        "{\n"
        '  "SentenceCitationList": [\n'
        "    {\n"
        '      "Sentence": "Your Sentence here.",\n'
        '      "Citation": ["1", "2"]\n'
        "    },\n"
        "    {\n"
        '      "Sentence": "Another Sentence here.",\n'
        '      "Citation": ["Figure 3"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Model Response:\n" + formatted_response + "\n\n"
        "Extracted Sentence-Citation pairs:\n"
    )
    return prompt


def evaluate_response(question: Dict, client: OpenAI) -> Dict:
    """
    使用 client.beta.chat.completions.parse(...) 对模型的回答进行评分，返回处理后的 question。
    若已存在 'sentence_citation_list' 则跳过。
    """
    if "sentence_citation_list" in question:
        return question

    response_text = question.get("response", "").strip()

    prompt = build_prompt(response_text)

    retries = 0
    while retries < MAX_RETRIES:
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an impartial judge."},
                    {"role": "user", "content": prompt},
                ],
                response_format=SentenceCitationList,
                temperature=0,
                max_completion_tokens=4096,
            )
            if (
                hasattr(completion.choices[0].message, "refusal")
                and completion.choices[0].message.refusal
            ):
                logging.info(f"Refusal for question {question.get('question_id')}.")
                question["sentence_citation_list"] = []
            else:
                parsed = completion.choices[0].message.parsed
                question["sentence_citation_list"] = [
                    pair.model_dump() for pair in parsed.SentenceCitationList
                ]

            return question
        except (
            RequestException,
        ) as e:
            logging.error(f"API error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
            retries += 1
            time.sleep(BACKOFF_FACTOR**retries)
        except ValidationError as ve:
            logging.error(f"Validation error during parsing: {ve}")
            question["sentence_citation_list"] = []
            return question
        except Exception as e:
            logging.error(f"Unexpected error during evaluation: {e}")
            question["sentence_citation_list"] = []
            return question

    logging.error(f"Failed to get rating after {MAX_RETRIES} retries.")
    question["sentence_citation_list"] = []
    return question


def process_jsonl_file(input_file: str, output_file: str, client: OpenAI):
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
                    item["sentence_citation_list"] = []
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
    """
    遍历 input_dir 目录下的所有 JSONL 文件，并对每个文件应用 process_jsonl_file 函数。
    """
    vqa_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(vqa_path.glob("*.jsonl"))
    if not jsonl_files:
        logging.warning(f"No JSONL found in {input_dir}")
        print(f"No JSONL found in {input_dir}")
        return

    for jfile in jsonl_files:
        out_file = output_path / jfile.name
        print(f"Processing: {jfile} -> {out_file}")
        logging.info(f"Processing {jfile} -> {out_file}")
        process_jsonl_file(str(jfile), str(out_file), client)


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    process_all_jsonl_in_directory(
        input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, client=client
    )


if __name__ == "__main__":
    main()
