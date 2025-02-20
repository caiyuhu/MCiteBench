import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional
from utils.utils import load_jsonl, save_jsonl
from typing import List, Tuple

INPUT_DIR = "../../output/entailment_judge"
OUTPUT_DIR = "../../output/citation_f1_source_f1_em"
CSV_OUTPUT = "../../output/citation_f1_source_f1_em/citation_f1_source_f1_em.csv"

# Specify the desired model order; empty strings indicate a blank line separation.
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


###################################
# Calculation Functions (Recall, Precision, F1)
###################################
def compute_cit_recall(recall_list: List[Dict[str, int]]) -> Optional[float]:
    valid_scores = []
    for d in recall_list:
        if d == {}:
            continue
        score = d.get("score")
        if score in (0, 1, 2):
            valid_scores.append(score)
    if not valid_scores:
        return None
    avg_score = sum(valid_scores) / len(valid_scores)
    return (avg_score / 2) * 100

def compute_cit_precision(precision_list: List[List[Dict[str, int]]]) -> Optional[float]:
    sentence_precisions = []
    for sub in precision_list:
        if not sub:
            continue
        valid_scores = []
        for d in sub:
            score = d.get("score")
            if score in (0, 1):
                valid_scores.append(score)
        if valid_scores:
            avg_sub = sum(valid_scores) / len(valid_scores)
            sentence_precisions.append(avg_sub)
    if not sentence_precisions:
        return None
    mean_precision = sum(sentence_precisions) / len(sentence_precisions)
    return mean_precision * 100

def compute_f1(r: Optional[float], p: Optional[float]) -> Optional[float]:
    if r is None or p is None:
        return None
    r /= 100
    p /= 100
    if (r + p) == 0:
        return 0.0
    f1 = 2 * r * p / (r + p)
    return f1 * 100

###################################
# New source Metrics Functions
###################################
def compute_source_metrics(ground_truth_list: List[str],
                          prediction_list: List[str]) -> Tuple[float, float, float]:
    gt_set = set(ground_truth_list)
    pred_set = set(prediction_list)
    if not gt_set and not pred_set:
        return 0.0, 0.0, 0.0
    correct = len(gt_set.intersection(pred_set))
    recall = (correct / len(gt_set) * 100) if gt_set else 0.0
    precision = (correct / len(pred_set) * 100) if pred_set else 0.0
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1

def compute_source_metrics_exact(ground_truth_list: List[str],
                                prediction_list: List[str]) -> Tuple[float, float, float]:
    gt_set = set(ground_truth_list)
    pred_set = set(prediction_list)
    if gt_set and gt_set == pred_set:
        return 100.0, 100.0, 100.0
    else:
        return 0.0, 0.0, 0.0

###################################
# Ground Truth Extraction Helpers
###################################
figure_pattern = re.compile(r"figure\s*\d+", re.IGNORECASE)
table_pattern = re.compile(r"table\s*\d+", re.IGNORECASE)

def get_ground_truth_list(rec: Dict) -> List[str]:
    evidence_keys = rec.get("evidence_keys", [])
    evidence_contents = rec.get("evidence_contents", [])
    text_2_idx = rec.get("text_2_idx", {})

    ground_truth_list = []
    for ek, ec in zip(evidence_keys, evidence_contents):
        if ec.startswith("images/"):
            fig_match = figure_pattern.search(ek)
            tab_match = table_pattern.search(ek)
            if fig_match:
                ground_truth_list.append(fig_match.group(0))
            elif tab_match:
                ground_truth_list.append(tab_match.group(0))
        else:
            if ec in text_2_idx:
                ground_truth_list.append(text_2_idx[ec])
    return ground_truth_list

def get_prediction_list(rec: Dict) -> List[str]:
    prediction_list = []
    sc_list = rec.get("sentence_citation_list", [])
    for item in sc_list:
        cits = item.get("Citation", [])
        if isinstance(cits, list):
            prediction_list.extend(cits)
    return prediction_list

###################################
# Process Single Record
###################################
def process_record(rec: Dict) -> Optional[Dict]:
    recall_list = rec.get("recall_list", [])
    precision_list = rec.get("precision_list", [])
    if not recall_list and not precision_list:
        return None

    cit_recall = compute_cit_recall(recall_list)
    cit_precision = compute_cit_precision(precision_list)
    cit_f1 = compute_f1(cit_recall, cit_precision)
    if cit_recall is None and cit_precision is None:
        return None

    rec["cit_recall"] = cit_recall
    rec["cit_precision"] = cit_precision
    rec["cit_f1"] = cit_f1

    ground_truth_list = get_ground_truth_list(rec)
    prediction_list = get_prediction_list(rec)

    source_recall, source_precision, source_f1 = compute_source_metrics(ground_truth_list, prediction_list)
    rec["source_recall"] = source_recall
    rec["source_precision"] = source_precision
    rec["source_f1"] = source_f1

    source_recall_em, source_precision_em, source_f1_em = compute_source_metrics_exact(ground_truth_list, prediction_list)
    rec["source_recall_em"] = source_recall_em
    rec["source_precision_em"] = source_precision_em
    rec["source_f1_em"] = source_f1_em

    return rec

###################################
# Aggregator & Aggregation Functions
###################################
def add_score(lst, r, p, f):
    if f is not None:
        lst.append((r, p, f))

def mean_rpf(lst: list) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not lst:
        return None, None, None
    rr = [x[0] for x in lst]
    pp = [x[1] for x in lst]
    ff = [x[2] for x in lst]
    n = len(lst)
    mean_r = sum(rr) / n if rr else None
    mean_p = sum(pp) / n if pp else None
    mean_f = sum(ff) / n if ff else None
    return mean_r, mean_p, mean_f

def mean_val(lst: List[tuple]) -> Optional[float]:
    if not lst:
        return None
    values = [x[0] for x in lst]
    return sum(values) / len(values)

###################################
# Process Folder and Write Output
###################################
def process_folder(input_dir: str, output_dir: str, csv_output: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_rows = []
    all_files = list(in_path.glob("*.jsonl"))
    for f in all_files:
        model_name = f.stem
        data = load_jsonl(str(f))
        processed_data = []

        aggregator = {
            "explanation_single": [],
            "explanation_single_source": [],
            "explanation_single_source_em": [],
            "explanation_multi": [],
            "explanation_multi_source": [],
            "explanation_multi_source_em": [],
            "locating_single": [],
            "locating_single_source": [],
            "locating_single_source_em": []
        }

        for rec in data:
            new_rec = process_record(rec)
            if not new_rec:
                continue
            processed_data.append(new_rec)

            r = new_rec["cit_f1"]
            p = new_rec["cit_precision"]
            f_score = new_rec["cit_f1"]

            ecount = new_rec.get("evidence_count", 0)
            qtype = new_rec.get("question_type", "").lower()
            if qtype == "explanation":
                if ecount == 1:
                    add_score(aggregator["explanation_single"], r, p, f_score)
                    aggregator["explanation_single_source"].append((new_rec["source_f1"],))
                    aggregator["explanation_single_source_em"].append((new_rec["source_f1_em"],))
                elif ecount > 1:
                    add_score(aggregator["explanation_multi"], r, p, f_score)
                    aggregator["explanation_multi_source"].append((new_rec["source_f1"],))
                    aggregator["explanation_multi_source_em"].append((new_rec["source_f1_em"],))
            elif qtype == "locating":
                if ecount == 1:
                    add_score(aggregator["locating_single"], r, p, f_score)
                    aggregator["locating_single_source"].append((new_rec["source_f1"],))
                    aggregator["locating_single_source_em"].append((new_rec["source_f1_em"],))

        out_file = out_path / f.name
        save_jsonl(processed_data, str(out_file))

        explanation_single_cit_f1 = mean_rpf(aggregator["explanation_single"])[2]
        explanation_single_source_f1 = mean_val(aggregator["explanation_single_source"])
        explanation_single_source_em = mean_val(aggregator["explanation_single_source_em"])

        explanation_multi_cit_f1 = mean_rpf(aggregator["explanation_multi"])[2]
        explanation_multi_source_f1 = mean_val(aggregator["explanation_multi_source"])
        explanation_multi_source_em = mean_val(aggregator["explanation_multi_source_em"])

        locating_single_cit_f1 = mean_rpf(aggregator["locating_single"])[2]
        locating_single_source_f1 = mean_val(aggregator["locating_single_source"])
        locating_single_source_em = mean_val(aggregator["locating_single_source_em"])

        valid_count = len(processed_data)

        row = {
            "model": model_name,
            "explanation_single_cit_f1": f"{explanation_single_cit_f1:.2f}" if explanation_single_cit_f1 is not None else "NA",
            "explanation_single_source_f1": f"{explanation_single_source_f1:.2f}" if explanation_single_source_f1 is not None else "NA",
            "explanation_single_source_em": f"{explanation_single_source_em:.2f}" if explanation_single_source_em is not None else "NA",
            "~": "",
            "explanation_multi_cit_f1": f"{explanation_multi_cit_f1:.2f}" if explanation_multi_cit_f1 is not None else "NA",
            "explanation_multi_source_f1": f"{explanation_multi_source_f1:.2f}" if explanation_multi_source_f1 is not None else "NA",
            "explanation_multi_source_em": f"{explanation_multi_source_em:.2f}" if explanation_multi_source_em is not None else "NA",
            "~": "",
            "locating_single_cit_f1": f"{locating_single_cit_f1:.2f}" if locating_single_cit_f1 is not None else "NA",
            "locating_single_source_f1": f"{locating_single_source_f1:.2f}" if locating_single_source_f1 is not None else "NA",
            "locating_single_source_em": f"{locating_single_source_em:.2f}" if locating_single_source_em is not None else "NA",
            "~": "",
            "valid_count": str(valid_count)
        }
        model_rows.append(row)

    ordered_rows = []
    model_dict = {row["model"]: row for row in model_rows}
    print(model_dict)
    for model in ordered_models:
        if model == "":
            ordered_rows.append({key: "" for key in model_rows[0].keys()})
        elif model in model_dict:
            ordered_rows.append(model_dict[model])

    fieldnames = [
        "model",
        "explanation_single_cit_f1",
        "explanation_single_source_f1",
        "explanation_single_source_em",
        "~",
        "explanation_multi_cit_f1",
        "explanation_multi_source_f1",
        "explanation_multi_source_em",
        "~",
        "locating_single_cit_f1",
        "locating_single_source_f1",
        "locating_single_source_em",
        "~",
        "valid_count"
    ]
    
    with open(csv_output, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in ordered_rows:
            writer.writerow(r)
    print(f"[INFO] CSV saved => {csv_output}")

if __name__=="__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR, CSV_OUTPUT)