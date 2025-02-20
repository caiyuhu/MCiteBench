import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import argparse
import json

from pathlib import Path

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from utils.utils import load_config, process_batch, remove_image_fields

config = load_config("../configs/config.json")
VISUAL_RESOURCES_DIR = f"../data/{config['VISUAL_RESOURCES_DIR']}"
INPUT_QUESTION = config["INPUT_QUESTION"]

def generate_via_local_vlm_mllama(args):
    model = args.model
    input_path = args.input_path
    output_dir = f"../output/responses"
    os.makedirs(output_dir, exist_ok=True)
    # if a local model path is provided, or just a model name
    model_name = Path(model).name if '/' in model else model
    output_file = os.path.join(output_dir, f"{model_name}.jsonl")
    sys_prompt_path = args.sys_prompt_path

    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    questions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    processed_question_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                processed_question_ids.add(json.loads(line)["question_id"])

    vlm = MllamaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        do_sample=False,
    )
    print(f"Pipeline initialized for {model_name}.")
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model)

    batch_inputs = []
    for question in questions:
        if question["question_id"] in processed_question_ids:
            continue

        pdf_id = question.get("pdf_id", "").replace(".pdf", "")
        idx_2_text = question.get("idx_2_text", {})
        evidence_list = question.get("evidence_contents", [])
        distractor_list = question.get("distractor_contents", [])

        combined_raw = evidence_list + distractor_list
        text_items, image_items = [], []

        for item in combined_raw:
            if isinstance(item, str) and item.startswith("images/"):
                image_items.append(item)
            else:
                text_items.append(item)

        combined_info = text_items + image_items
        content_list = []
        placeholders_for_text = 0

        # put text/image into content_list
        for c_item in combined_info:
            if isinstance(c_item, str) and c_item.startswith("images/"):
                full_img_path = Path(VISUAL_RESOURCES_DIR) / pdf_id / c_item
                if not full_img_path.exists():
                    print(f"Warning: Image not found at {full_img_path}. Skip.")
                    continue
                content_list.append({"type": "image", "image": str(full_img_path)})
            else:
                content_list.append(None)
                placeholders_for_text += 1

        # sort idx_2_text keys and insert text
        sorted_keys = sorted(map(int, idx_2_text.keys()))
        if len(sorted_keys) != placeholders_for_text:
            print(
                f"Warning: text count mismatch for question {question['question_id']}."
            )

        text_idx_pos = 0
        for i in range(len(content_list)):
            if content_list[i] is None:
                if text_idx_pos >= len(sorted_keys):
                    content_list[i] = {"type": "text", "text": "[?] MISSING"}
                    continue
                key = sorted_keys[text_idx_pos]
                text_val = idx_2_text[str(key)].strip()
                content_list[i] = {"type": "text", "text": f"[{key}] {text_val}"}
                text_idx_pos += 1

        q_text = question.get("Question", question.get("question"))
        question_prompt = f"Question:\n{q_text}"

        messages = []
        messages.append({"role": "user", "content": content_list})
        messages.append({"role": "user", "content": question_prompt})
        messages.append({"role": "user", "content": sys_prompt})

        batch_inputs.append((messages, question))

    if not batch_inputs:
        print(f"No new questions to process for model {model_name}.")
        return

    batch_size = 1
    with open(output_file, "a", encoding="utf-8") as output_f:
        for i in range(0, len(batch_inputs), batch_size):
            current_batch = batch_inputs[i : i + batch_size]
            # get the messages from each (messages, question) pair
            prompts = [data[0] for data in current_batch]

            # generate a 2D image_list with the same shape as prompts
            image_list = process_batch(prompts)

            # remove image fields from prompts
            prompts_no_images = [remove_image_fields(prompt) for prompt in prompts]

            texts = [
                processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts_no_images
            ]

            if image_list == [[]]:
                inputs = processor(
                    text=texts,
                    return_tensors="pt",
                ).to(vlm.device)
            else:
                inputs = processor(
                    text=texts,
                    images=image_list,
                    return_tensors="pt",
                ).to(vlm.device)

            generated_ids = vlm.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for j, response in enumerate(output_texts):
                _, question = current_batch[j]
                question["response"] = response
                output_f.write(json.dumps(question, ensure_ascii=False) + "\n")
                output_f.flush()

            print(f"Processed batch {i // batch_size + 1} for model {model_name}.")

    print(f"Generated answers for model {model_name} saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use local vlm to process multimodal input."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Llama-3.2-11B-Vision-Instruct",
        help="The name of the local model to use.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=f"../data/{INPUT_QUESTION}.jsonl",
        help="Path to the input questions JSONL file.",
    )
    parser.add_argument(
        "--sys_prompt_path",
        type=str,
        default="../prompts/response_with_citation_prompt.txt",
        help="Path to the system prompt file.",
    )
    args = parser.parse_args()

    generate_via_local_vlm_mllama(args)
