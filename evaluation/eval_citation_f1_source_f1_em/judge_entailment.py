import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import openai
from openai import OpenAI
from pydantic import BaseModel
from requests.exceptions import RequestException
from tqdm import tqdm

from utils.utils import load_config, load_jsonl, encode_image

config = load_config("../../configs/config.json")
OPENAI_API_KEY = config["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


MODEL_NAME = "gpt-4o-2024-11-20"

INPUT_DIR = "../../output/extracted_sen_cit_list"
OUTPUT_DIR = "../../output/entailment_judge"
VISUAL_RESOURCES_DIR = f"../../data/{config["VISUAL_RESOURCES_DIR"]}"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2
MAX_WORKERS = 8

class Rating(BaseModel):
    rating: int

logging.basicConfig(
    filename="extract_response_cit_f1.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def extract_index_from_str(citation_str: str) -> Optional[int]:
    if citation_str.isdigit():
        return int(citation_str)
    return None


def extract_figure_table_index(citation_str: str) -> Optional[int]:
    match_fig = re.match(r"^(?:figure|fig)\s*(\d+)$", citation_str, re.IGNORECASE)
    if match_fig:
        return int(match_fig.group(1))
    match_tab = re.match(r"^(?:table|tab)\s*(\d+)$", citation_str, re.IGNORECASE)
    if match_tab:
        return int(match_tab.group(1))
    return None


def build_content_list_for_citation(
    citation_list: List[str],
    pdf_id: str,
    idx_2_text: Dict[str, str],
    idx_2_image: Dict[str, str],
    idx_2_table: Dict[str, str],
    VISUAL_RESOURCES_DIR: str,
) -> Optional[List[Dict]]:
    content_list: List[Dict] = []

    for cit_str in citation_list:
        # 1) if it's a pure number, then it's a text reference
        idx_num = extract_index_from_str(cit_str)
        if idx_num is not None:
            idx_key = str(idx_num)
            if idx_key not in idx_2_text:
                continue
            text_snippet = idx_2_text[idx_key]
            content_list.append({"type": "text", "text": text_snippet})
            continue

        # 2) figure/table
        ft_idx = extract_figure_table_index(
            cit_str
        )
        if ft_idx is not None:
            ft_idx = str(ft_idx)
            clow = cit_str.lower()
            if "fig" in clow:
                if ft_idx not in idx_2_image:
                    continue
                rel_path = idx_2_image[ft_idx]
                full_img_path = Path(VISUAL_RESOURCES_DIR) / pdf_id / rel_path
                if not full_img_path.exists():
                    continue
                encoded_str = encode_image(str(full_img_path))
                if not encoded_str:
                    continue
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_str}"},
                    }
                )
            elif "tab" in clow:
                if ft_idx not in idx_2_table:
                    continue
                rel_path = idx_2_table[ft_idx]
                full_table_path = Path(VISUAL_RESOURCES_DIR) / pdf_id / rel_path
                if not full_table_path.exists():
                    continue
                encoded_str = encode_image(str(full_table_path))
                if not encoded_str:
                    continue
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_str}"},
                    }
                )
            else:
                continue
            continue
        
        continue

    if not content_list:
        return None

    return content_list


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON decode error in {file_path}: line {line_number}, {e}"
                )
    return data


def save_jsonl(data: List[Dict], file_path: str):
    out_dir = Path(file_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def extract_index_from_str(citation_str: str) -> Optional[int]:
    if citation_str.isdigit():
        return int(citation_str)
    return None


def extract_figure_table_index(citation_str: str) -> Optional[int]:
    match_fig = re.match(r"^(?:figure|fig)\s*(\d+)$", citation_str, re.IGNORECASE)
    if match_fig:
        return int(match_fig.group(1))
    match_tab = re.match(r"^(?:table|tab)\s*(\d+)$", citation_str, re.IGNORECASE)
    if match_tab:
        return int(match_tab.group(1))
    return None


def build_recall_prompt(sentence: str) -> str:
    prompt = (
        "You are an expert in evaluating text quality. You will receive a statement from an AI assistant’s response based on a paper, along with a part from the document (which could be a text paragraph, image, or table). Your task is to carefully assess whether this statement is supported by the provided part. Please use the following scale to generate your rating:"
        "0: No support — The statement is largely unrelated to the provided part (text, image, or table), or most key points in the statement do not align with the content of the part."
        "1: Partially supported — More than half of the content in the statement is supported by the part, but a small portion is either not mentioned or contradicts the part."
        "2: Fully supported — Most information in the statement is supported by or extracted from the part. This applies only to cases where the statement and the part are almost identical."
        "Ensure that you do not use any information or knowledge outside of the provided part when evaluating. Please return only the rating in JSON format, with 0, 1, or 2."
        "Statement: " + sentence
    )
    return prompt


def build_precision_prompt(sentence: str) -> str:
    prompt = (
        "You are an expert in evaluating text quality. You will receive a statement from an AI assistant’s response based on a paper, along with a part from the document (which could be a text paragraph, image, or table). Your task is to carefully assess whether the provided part contains some key information of the statement. Please use the following scale to generate your rating:"
        "0: Unrelevant — The statement is almost unrelated to the provided part, or all key points of the statement are inconsistent with the the provided part."
        "1: Relevant — Some key points of the statement are supported by or extracted from the the provided part."
        "Ensure that you do not use any information or knowledge outside of the provided part when evaluating. Please return only the rating in JSON format, with 0 or 1."
        "Statement: " + sentence
    )
    return prompt


def evaluate_implied_relationship(
    mode: str,
    sentence: str,
    cits: List[str],
    pdf_id: str,
    idx_2_text: Dict[str, str],
    idx_2_image: Dict[str, str],
    idx_2_table: Dict[str, str],
    client: OpenAI,
    content_list: Optional[List[Dict]] = None,
) -> int:
    if content_list is None:
        content_list = build_content_list_for_citation(
            cits, pdf_id, idx_2_text, idx_2_image, idx_2_table, VISUAL_RESOURCES_DIR
        )
    if content_list is None or len(content_list) == 0:
        return 0

    if mode == "recall":
        prompt_for_llm = build_recall_prompt(sentence)
    elif mode == "precision":
        prompt_for_llm = build_precision_prompt(sentence)
    messages = [
        {"role": "system", "content": prompt_for_llm},
        {"role": "user", "content": content_list},
    ]

    retries = 0
    while retries < MAX_RETRIES:
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=messages,
                response_format=Rating,
                temperature=0,
            )
            if (
                hasattr(completion.choices[0].message, "refusal")
                and completion.choices[0].message.refusal
            ):
                logging.info(f"Refusal for sentence {sentence}.")
                rating = 0
            else:
                parsed = completion.choices[0].message.parsed
                rating = parsed.rating

            return rating
        except (
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.Timeout,
            RequestException,
        ) as e:
            logging.error(f"API error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
            retries += 1
            time.sleep(BACKOFF_FACTOR**retries)
        except Exception as e:
            logging.error(f"Unexpected error during evaluation: {e}")
            rating = 0
            return rating

    logging.error(f"Failed to get rating after {MAX_RETRIES} retries.")
    rating = 0
    return rating


def check_citation_exists_single(
    cit_str: str,
    pdf_id: str,
    idx_2_text: Dict[str, str],
    idx_2_image: Dict[str, str],
    idx_2_table: Dict[str, str],
) -> bool:
    # 1) pure num => text
    idx_num = extract_index_from_str(cit_str)
    if idx_num is not None:
        if str(idx_num) in idx_2_text:
            return True
        else:
            return False
    # 2) figure/table
    ft_idx = extract_figure_table_index(cit_str)
    if ft_idx is not None:
        ft_idx = str(ft_idx)
        clow = cit_str.lower()
        if "fig" in clow:
            if ft_idx in idx_2_image:
                return True
        elif "tab" in clow:
            if ft_idx in idx_2_table:
                return True
        return False

    return False


def get_recall_score(
    sentence: str,
    cits: List[str],
    pdf_id: str,
    idx_2_text: Dict[str, str],
    idx_2_image: Dict[str, str],
    idx_2_table: Dict[str, str],
    client: OpenAI,
) -> int:
    content_list = build_content_list_for_citation(
        cits, pdf_id, idx_2_text, idx_2_image, idx_2_table, VISUAL_RESOURCES_DIR
    )
    if content_list is None or len(content_list) == 0:
        return 0

    score = evaluate_implied_relationship(
        "recall",
        sentence,
        cits,
        pdf_id,
        idx_2_text,
        idx_2_image,
        idx_2_table,
        client,
        content_list=content_list,
    )
    return score


def get_precision_score(
    sentence: str,
    cit: str,
    pdf_id: str,
    idx_2_text: Dict[str, str],
    idx_2_image: Dict[str, str],
    idx_2_table: Dict[str, str],
    client: OpenAI,
) -> int:
    content_list = build_content_list_for_citation(
        [cit], pdf_id, idx_2_text, idx_2_image, idx_2_table, VISUAL_RESOURCES_DIR
    )
    if content_list is None or len(content_list) == 0:
        return 0
    else:
        score = evaluate_implied_relationship(
            "precision",
            sentence,
            [cit],
            pdf_id,
            idx_2_text,
            idx_2_image,
            idx_2_table,
            client,
            content_list=content_list,
        )
    return score


def process_json_entry(entry: Dict, client: OpenAI) -> Dict:
    # 如果已有 recall_list 且已有 precision_list，直接跳过
    if "recall_list" in entry and "precision_list" in entry:
        return entry  # 不做任何处理

    pdf_id = entry.get("pdf_id", "")
    pdf_id = pdf_id.replace(".pdf", "")

    idx_2_text = entry.get("idx_2_text", {})
    idx_2_image = entry.get("idx_2_image", {})
    idx_2_table = entry.get("idx_2_table", {})

    sentence_citation_list = entry.get("sentence_citation_list", [])

    if not sentence_citation_list:
        entry["recall_list"] = []
        entry["precision_list"] = []
        return entry

    recall_list = []
    precision_list = []
    for sc_pair in sentence_citation_list:
        sentence = sc_pair.get("Sentence", "")
        cits = sc_pair.get("Citation", [])
        if not cits:
            recall_list.append({})
            precision_list.append([])
            continue

        found_cits = []
        for cit in cits:
            if check_citation_exists_single(
                cit, pdf_id, idx_2_text, idx_2_image, idx_2_table
            ):
                found_cits.append(cit)

        if not found_cits:
            recall_list.append({"score": 0})
            precision_scores = [{"score": 0} for _ in cits]
            precision_list.append(precision_scores)
            continue
        else:
            recall_score = get_recall_score(
                sentence,
                found_cits,
                pdf_id,
                idx_2_text,
                idx_2_image,
                idx_2_table,
                client,
            )
            recall_list.append({"score": recall_score})

            p_scores = []
            for cit in cits:
                if check_citation_exists_single(
                    cit, pdf_id, idx_2_text, idx_2_image, idx_2_table
                ):
                    score = get_precision_score(
                        sentence,
                        cit,
                        pdf_id,
                        idx_2_text,
                        idx_2_image,
                        idx_2_table,
                        client,
                    )
                    p_scores.append({"score": score})
                else:
                    p_scores.append({"score": 0})
            precision_list.append(p_scores)

    entry["recall_list"] = recall_list
    entry["precision_list"] = precision_list
    return entry

def process_one_jsonl_file(input_file: str, output_file: str, client: OpenAI):
    data = load_jsonl(input_file)
    original_count = len(data)

    if os.path.exists(output_file):
        existing_data = load_jsonl(output_file)
        existing_ids = {
            x.get("question_id") for x in existing_data if "question_id" in x
        }
        data_to_process = [d for d in data if d.get("question_id") not in existing_ids]
        duplicates_skipped = original_count - len(data_to_process)
        logging.info(
            f"[{input_file}] skip {duplicates_skipped} duplicates by question_id."
        )
    else:
        data_to_process = data

    processed_count = 0
    skipped_entries = 0

    if not data_to_process:
        logging.info(f"[{input_file}] No new items to process.")
        return


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(
        output_file, "a", encoding="utf-8"
    ) as f, tqdm(total=len(data_to_process), desc="Processing items") as pbar:
        future_to_item = {
            executor.submit(process_json_entry, item, client): item
            for item in data_to_process
        }

        for fut in as_completed(future_to_item):
            item = future_to_item[fut]
            try:
                ret = fut.result()
                f.write(json.dumps(ret, ensure_ascii=False) + "\n")
                f.flush()
                processed_count += 1
            except Exception as e:
                logging.error(f"Error processing item {item.get('question_id')}: {e}")
                item["recall_list"] = []
                item["precision_list"] = []
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                skipped_entries += 1
            finally:
                pbar.update(1)

    logging.info(
        f"[{input_file}] done: total={original_count}, processed={processed_count}, skipped_err={skipped_entries}"
    )


def process_all_files(input_dir: str, output_dir: str, client: OpenAI):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_jsonl = list(in_path.glob("*.jsonl"))
    if not all_jsonl:
        print(f"[WARNING] No .jsonl in {input_dir}")
        return

    for jfile in all_jsonl:
        print(f"Processing file: {jfile}")
        out_f = out_path / jfile.name
        logging.info(f"Start processing {jfile} -> {out_f}")
        process_one_jsonl_file(str(jfile), str(out_f), client)


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    process_all_files(INPUT_DIR, OUTPUT_DIR, client)


if __name__ == "__main__":
    main()
