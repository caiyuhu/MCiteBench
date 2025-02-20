**Note:**  
Before running the evaluation, please ensure that you have followed the instructions in `MCiteBench/generation/README.md` to generate the responses.

# 1. Answer Accuracy

## LLM Scoring

```python
cd evaluation/eval_ans_acc
python calculate_ans_acc_score.py
```

After execution, the results will be saved in `MCiteBench/output/answer_acc`. Each JSONL file will contain an additional `answer_rating` field, representing the LLM’s score for each response.

## Score Calculation

```python
cd evaluation/eval_ans_acc
python calculate_ans_acc_score.py
```

This will create a `MCiteBench/output/answer_acc/ans_acc_scores.csv` file containing the accuracy scores.

---

# 2. Citation F1

## Splitting Sentence Citations

```
cd evaluation/eval_citation_f1_source_f1_em
python extract_sen_cit_list.py
```

After execution, the `MCiteBench/output/extracted_sen_cit_list` folder will be created. Each JSONL file within will include a `sentence_citation_list field, which contains the split sentences and their corresponding citations.

```json
{
	"sentence_citation_list": [
          {"Sentence": "For example, in Table 4, the \"InternVL2-8b\" model achieves high performance on the DocVQA dataset with 79.48% precision (P) and 47.50% consistency (C).", "Citation": ["Table 4"]},
  	  {"Sentence": "Additionally, Table 6 demonstrates that C&P fine-tuning enhances both cognitive task (C.T.) and perceptual task (P.T.) performance.", "Citation": ["Table 6"]}
        ]
}
```

## Entailment Judgment

This step should only be run after completing the sentence citation splitting process.

```python
cd evaluation/eval_citation_f1_source_f1_em
python judge_entailment.py
```

After execution, `MCiteBench/output/entailment_judge` will contain the resulting files. Each JSONL file will include recall_list and precision_list fields, with scores corresponding to the sentence-citation pairs in the `sentence_citation_list`.

```json
{
	"sentence_citation_list": [
          {"Sentence": "For example, in Table 4, the \"InternVL2-8b\" model achieves high performance on the DocVQA dataset with 79.48% precision (P) and 47.50% consistency (C).", "Citation": ["Table 4"]},
  	  {"Sentence": "Additionally, Table 6 demonstrates that C&P fine-tuning enhances both cognitive task (C.T.) and perceptual task (P.T.) performance.", "Citation": ["Table 6"]}
        ]
        "recall_list": [{"score": 2}, {"score": 2}], 
        "precision_list": [[{"score": 1}], [{"score": 1}]]
}
```

The scoring for Citation F1/Source F1/Source EM will be calculated together, as outlined below.

---

# 3. Source F1/Source EM

## Score Calculation

```python
cd evaluation/eval_citation_f1_source_f1_em
python calculate_citation_source_score.py
```

After execution, the `MCiteBench/output/citation_f1_source_f1_em/` folder will be created. This folder will contain the `citation_f1_source_f1_em.csv` file, as well as the modified JSONL files. Specifically, the following fields will be added to each JSONL file to represent key statistics for each response:

```json
{
  "cit_recall": 100.0, 
  "cit_precision": 50.0, 
  "cit_f1": 66.66666666666666, 
  "source_recall": 50.0, 
  "source_precision": 50.0, 
  "source_f1": 50.0, 
  "source_recall_em": 0.0, 
  "source_precision_em": 0.0, 
  "source_f1_em": 0.0
}
```

