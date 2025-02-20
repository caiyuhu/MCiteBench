import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from utils.utils import encode_image, load_config

config = load_config("../configs/config.json")
OPENAI_API_KEY = config["OPENAI_API_KEY"]
VISUAL_RESOURCES_DIR = f"../data/{config['VISUAL_RESOURCES_DIR']}"
INPUT_QUESTION = config["INPUT_QUESTION"]


def generate_via_openai_batch(batch, sys_prompt, model, output_file, client):

    with open(output_file, "a", encoding="utf-8") as output_f:
        for question in batch:
            try:
                q_text = question.get("Question")
                if not q_text:
                    q_text = question.get("question")

                pdf_id = question.get("pdf_id", "")

                pdf_id = pdf_id.replace(".pdf", "")
                idx_2_text = question.get("idx_2_text", {})

                evidence_list = question.get("evidence_contents", [])
                distractor_list = question.get("distractor_contents", [])

                # 1) Combine evidence and distractor, then shuffle
                combined_info = list(evidence_list + distractor_list)
                random.seed(42)
                random.shuffle(combined_info)

                # 2) First loop: put images in, leave placeholders for text
                content_list = []
                placeholders_for_text = 0

                for item in combined_info:
                    # if item is an image path, encode it and put in content_list
                    if isinstance(item, str) and item.startswith("images/"):
                        full_img_path = Path(VISUAL_RESOURCES_DIR) / pdf_id / item
                        if not full_img_path.exists():
                            print(f"Warning: Image not found at {full_img_path}. Skip.")
                            continue
                        img_base64 = encode_image(str(full_img_path))
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            }
                        )
                    else:
                        # else treat as text => put a placeholder (to be filled later)
                        content_list.append(None)
                        placeholders_for_text += 1

                # 3) Sort the keys of idx_2_text in ascending order
                sorted_keys = sorted(map(int, idx_2_text.keys()))

                if len(sorted_keys) != placeholders_for_text:
                    print(
                        f"Warning: text count mismatch for question {question.get('question_id')}."
                    )

                # 4) Second loop: fill in the placeholders with text
                text_idx_pos = 0
                for i in range(len(content_list)):
                    if content_list[i] is None:
                        if text_idx_pos >= len(sorted_keys):
                            content_list[i] = {"type": "text", "text": "[?] MISSING"}
                            continue

                        key = sorted_keys[text_idx_pos]
                        text_val = idx_2_text[str(key)].strip()
                        content_list[i] = {
                            "type": "text",
                            "text": f"[{key}] {text_val}",
                        }
                        text_idx_pos += 1

                question_prompt = f"Question:\n{q_text}"

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content_list},
                    {"role": "user", "content": question_prompt},
                ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=1024,
                    temperature=0,
                )
                answer = response.choices[0].message.content.strip()

                question["response"] = answer
                output_f.write(json.dumps(question, ensure_ascii=False) + "\n")
                output_f.flush()

            except Exception as e:
                print(f"Error generating response: {e}")
                continue


def generate_via_openai(args):
    """
    Main entry point for generating responses.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)

    model = args.model
    input_questions_path = args.input_questions_path

    sys_prompt_path = "../prompts/response_with_citation_prompt.txt"
    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    output_dir = f"../output/responses"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model}.jsonl")

    # if output_file exists, read all question_id, and skip questions that have been generated
    generated_question_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                question = json.loads(line.strip())
                generated_question_ids.add(question.get("question_id"))

    questions = []
    with open(input_questions_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            q_id = record.get("question_id")
            if q_id in generated_question_ids:
                print(f"Question {q_id} has been generated. Skip.")
                continue
            questions.append(record)

    print(f"Num of questions to process: {len(questions)}")

    batch_size = args.batch_size
    num_threads = args.num_threads

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            futures.append(
                executor.submit(
                    generate_via_openai_batch,
                    batch,
                    sys_prompt,
                    model,
                    output_file,
                    client,
                )
            )
        for f in as_completed(futures):
            f.result()
    print(f"Generation finished. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="The name of the OpenAI model to use.",
    )
    parser.add_argument(
        "--input_questions_path",
        type=str,
        default=f"../data/{INPUT_QUESTION}.jsonl",
        help="Path to the input questions JSONL file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size used for periodic saving.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads to use for parallel processing.",
    )
    args = parser.parse_args()

    generate_via_openai(args)
