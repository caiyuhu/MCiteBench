import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams

from utils.utils import encode_image, load_config

config = load_config("../configs/config.json")
VISUAL_RESOURCES_DIR = f"../data/{config['VISUAL_RESOURCES_DIR']}"
INPUT_QUESTION = config["INPUT_QUESTION"]



def get_llm(model, gpu_num):
    """Initialize the LLM based on model-specific configurations."""
    base_params = {
        "trust_remote_code": True,
        "model": model,
        "tensor_parallel_size": gpu_num,
        "limit_mm_per_prompt": {"image": 5},
    }
    return LLM(**base_params)


def generate_via_local_vlm(args):
    model = args.model
    input_questions_path = args.input_questions_path

    # Load system prompt
    sys_prompt_path = args.sys_prompt_path
    with open(sys_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    # Prepare output directory
    output_dir = f"../output/responses"
    os.makedirs(output_dir, exist_ok=True)
    # if a local model path is provided, or just a model name
    model_name = Path(model).name if '/' in model else model
    output_file = os.path.join(output_dir, f"{model_name}.jsonl")

    # Read input questions
    questions = []
    with open(input_questions_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    # Track processed question IDs
    processed_question_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                processed_question_ids.add(json.loads(line)["question_id"])

    # Initialize LLM
    llm = get_llm(model, args.gpu_num)
    print(f"Pipeline initialized for {model}.")

    # Prepare batch inputs
    batch_inputs = []
    for question in questions:
        if question["question_id"] in processed_question_ids:
            continue

        pdf_id = question.get("pdf_id", "").replace(".pdf", "")
        idx_2_text = question.get("idx_2_text", {})
        evidence_list = question.get("evidence_contents", [])
        distractor_list = question.get("distractor_contents", [])

        # Combine evidence and distractor lists
        combined_raw = evidence_list + distractor_list
        text_items = [item for item in combined_raw if not item.startswith("images/")]
        image_items = [item for item in combined_raw if item.startswith("images/")]
        combined_info = text_items + image_items

        # Process combined information
        content_list = []
        placeholders_for_text = 0

        for item in combined_info:
            if item.startswith("images/"):
                full_img_path = Path(VISUAL_RESOURCES_DIR) / pdf_id / item
                if not full_img_path.exists():
                    print(f"Warning: Image not found at {full_img_path}. Skip.")
                    continue
                img_base64 = encode_image(str(full_img_path))
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    }
                )
            else:
                content_list.append(None)
                placeholders_for_text += 1

        sorted_keys = sorted(map(int, idx_2_text.keys()))
        if len(sorted_keys) != placeholders_for_text:
            print(
                f"Warning: text count mismatch for question {question['question_id']}."
            )

        # Fill in text placeholders
        text_idx_pos = 0
        for i in range(len(content_list)):
            if content_list[i] is None:
                key = (
                    sorted_keys[text_idx_pos]
                    if text_idx_pos < len(sorted_keys)
                    else None
                )
                text_val = idx_2_text.get(str(key), "[?] MISSING").strip()
                content_list[i] = (
                    {"type": "text", "text": f"[{key}] {text_val}"}
                    if key
                    else {"type": "text", "text": "[?] MISSING"}
                )
                text_idx_pos += 1

        # Construct prompt
        q_text = question.get("Question") or question.get("question")
        question_prompt = f"Question:\n{q_text}"

        messages = [
            {
                "role": "system" if "llama" not in model.lower() else "user",
                "content": sys_prompt,
            },
            {"role": "user", "content": content_list},
            {"role": "user", "content": question_prompt},
        ]

        batch_inputs.append((messages, question))

    # Process batches
    if not batch_inputs:
        print(f"No new questions to process for model {model}.")
        return

    batch_size = args.batch_size
    with open(output_file, "a", encoding="utf-8") as output_f:
        for i in range(0, len(batch_inputs), batch_size):
            current_batch = batch_inputs[i : i + batch_size]
            prompts = [data[0] for data in current_batch]

            sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
            outputs = llm.chat(prompts, sampling_params=sampling_params)
            generated_text_list = [o.outputs[0].text for o in outputs]

            for j, response in enumerate(generated_text_list):
                _, question = current_batch[j]
                question["response"] = response
                output_f.write(json.dumps(question, ensure_ascii=False) + "\n")
                output_f.flush()

            print(f"Processed batch {i // batch_size + 1} for model {model}.")

    print(f"Generated answers for model {model} saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use local vlm to process multimodal input, text in front, images behind, but system & question at extremes."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2-VL-7B-Instruct",
        help="The name of the local model to use.",
    )
    parser.add_argument(
        "--input_questions_path",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used for parallel processing.",
    )
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use.")
    args = parser.parse_args()

    generate_via_local_vlm(args)
