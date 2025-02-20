import os
import json
import pandas as pd
from pathlib import Path

INPUT_FOLDER = "../../output/answer_acc"
OUTPUT_CSV = "../../output/answer_acc/ans_acc_scores.csv"

# ordered to match the output CSV
# Add modles or delete models as needed
ordered_models = [
    "llava-onevision-qwen2-7b-ov-hf",
    "llava-onevision-qwen2-7b-ov-chat-hf",
    "MiniCPM-V-2_6",
    "Qwen2-VL-7B-Instruct",
    "InternVL2_5-8B",
    "Llama-3.2-11B-Vision-Instruct",
    "",
    "Qwen2-VL-72B-Instruct",
    "InternVL2_5-78B",
    "Llama-3.2-90B-Vision-Instruct",
    "",
    "gpt-4o-mini",
    "gpt-4o-2024-11-20"
]

def add_rating(stats: dict, rating: float):
    stats["sum"] += rating
    stats["count"] += 1

def compute_acc(stats: dict) -> float:
    """
    According to stats => { 'sum': x, 'count': y }, compute accuracy:
        acc = (sum * 100) / (2 * count) ; return 0.0 if count is 0
    """
    if stats["count"] == 0:
        return 0.0
    return (stats["sum"] * 100.0) / (2.0 * stats["count"])

def process_file(jsonl_path: str) -> dict:
    """
    Read the single JSONL file and classify all records.
    In addition to overall statistics, the following are added:
        - records that are explanation and have evidence_count == 1 (explanation_single)
        - records that are explanation and have evidence_count > 1 (explanation_multi)
        - records that are locating and have evidence_count == 1 (locating_single)
    """
    aggregator = {
        "overall": {"sum": 0.0, "count": 0},
        "modal": {
            "figure": {"sum": 0.0, "count": 0},
            "table": {"sum": 0.0, "count": 0},
            "text": {"sum": 0.0, "count": 0},
            "mixed": {"sum": 0.0, "count": 0},
        },
        "single_source": {"sum": 0.0, "count": 0},
        "multi_source": {"sum": 0.0, "count": 0},
        "explanation": {"sum": 0.0, "count": 0},
        "locating": {"sum": 0.0, "count": 0},
        "subj_single": {"sum": 0.0, "count": 0},
        "subj_multi": {"sum": 0.0, "count": 0},
        "obj_single": {"sum": 0.0, "count": 0},
    }

    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                continue

            rating = data.get("answer_rating", None)
            if rating is None:
                continue

            add_rating(aggregator["overall"], rating)

            emodal = data.get("evidence_modal")
            if isinstance(emodal, list) and len(emodal) == 1:
                mod_str = emodal[0]
            elif isinstance(emodal, str):
                mod_str = emodal
            else:
                mod_str = None
            if mod_str == "mix":
                mod_str = "mixed"
            if mod_str in ("figure", "table", "text", "mixed"):
                add_rating(aggregator["modal"][mod_str], rating)

            # single_source / multi_source
            ecount = data.get("evidence_count", 0)
            if ecount == 1:
                add_rating(aggregator["single_source"], rating)
            elif ecount > 1:
                add_rating(aggregator["multi_source"], rating)

            # question_type
            qtype = data.get("question_type", "").lower()
            if qtype == "explanation":
                add_rating(aggregator["explanation"], rating)
            elif qtype == "locating":
                add_rating(aggregator["locating"], rating)

            if qtype == "explanation":
                if ecount == 1:
                    add_rating(aggregator["subj_single"], rating)
                elif ecount > 1:
                    add_rating(aggregator["subj_multi"], rating)
            elif qtype == "locating":
                if ecount == 1:
                    add_rating(aggregator["obj_single"], rating)

    return aggregator

def main():
    in_path = Path(INPUT_FOLDER)
    rows = []

    for filename in os.listdir(in_path):
        if not filename.endswith(".jsonl"):
            continue

        file_path = in_path / filename
        model_name = filename.replace(".jsonl", "").replace("generated_answers_", "")

        aggregator = process_file(str(file_path))

        subj_single_acc = compute_acc(aggregator["subj_single"])
        subj_multi_acc = compute_acc(aggregator["subj_multi"])
        obj_single_acc = compute_acc(aggregator["obj_single"])

        valid_count = aggregator["overall"]["count"]

        row = {
            "model": model_name,
            "explanation_single_acc": f"{subj_single_acc:.2f}",
            "explanation_multi_acc": f"{subj_multi_acc:.2f}",
            "locating_single_acc": f"{obj_single_acc:.2f}",
            "valid_count": str(valid_count)
        }

        rows.append(row)
        print(f"Model={model_name} explanation_single_acc={row['explanation_single_acc']} explanation_multi_acc={row['explanation_multi_acc']} locating_single_acc={row['locating_single_acc']} valid_count={valid_count}")

    ordered_rows = []
    columns = ["model", "explanation_single_acc", "explanation_multi_acc", "locating_single_acc", "valid_count"]

    for model in ordered_models:
        if model == "":
            empty_row = {col: "" for col in columns}
            ordered_rows.append(empty_row)
        else:
            found = next((r for r in rows if r["model"] == model), None)
            if found:
                ordered_rows.append(found)
            else:
                empty_row = {col: "" for col in columns}
                empty_row["model"] = model
                empty_row["valid_count"] = "0"
                ordered_rows.append(empty_row)

    if ordered_rows:
        df = pd.DataFrame(ordered_rows, columns=columns)
        out_csv_path = Path(OUTPUT_CSV)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_csv_path, index=False)
        print(f"[INFO] CSV saved => {out_csv_path}")
    else:
        print("[WARN] No model found or no valid data.")

if __name__ == "__main__":
    main()