## Data Download

We provide data samples in the `data` directory, which includes `data/visual_resources_example` and `data/data_example.jsonl`. You can run the entire pipeline with this sample data without needing the full dataset.

If you wish to use the full dataset, please download the [MCiteBench_full_dataset](https://drive.google.com/file/d/16zYXBMCk3h70sfrn7M28VrG3N96mXqEC/view?usp=drive_link) and extract it into the `data` directory. The full dataset will include `data/visual_resources` and `data/data.jsonl`.

Additionally, navigate to `configs/config.json` and modify the following fields:
-  Fill in your OpenAI API key
- "visual_resources_example" for test and "visual_resources" for full dataset
- "data_example" for test and "data" for full dataset
```json
{
    "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
    "VISUAL_RESOURCES_DIR": "visual_resources_example",
    "INPUT_QUESTION": "data_example"
}
```

## Data Format
The data format for `data_example.jsonl` and `data.jsonl` is as follows:

```yaml
question_id: [str]             # The ID of the question
pdf_id: [str]                  # The ID of the associated PDF document
question_type: [str]           # The type of question, with possible values: "explanation" or "locating"
question: [str]                # The text of the question
answer: [str]                  # The answer to the question, which can be a string, list, float, or integer, depending on the context

evidence_keys: [list]          # A list of abstract references or identifiers for evidence, such as "section x", "line y", "figure z", or "table k".
                               # These are not the actual content but pointers or descriptions indicating where the evidence can be found.
                               # Example: ["section 2.1", "line 45", "Figure 3"]
evidence_contents: [list]      # A list of resolved or actual evidence content corresponding to the `evidence_keys`.
                               # These can include text excerpts, image file paths, or table file paths that provide the actual evidence for the answer.
                               # Each item in this list corresponds directly to the same-index item in `evidence_keys`.
                               # Example: ["This is the content of section 2.1.", "/path/to/figure_3.jpg"]
evidence_modal: [str]          # The modality type of the evidence, with possible values: ['figure', 'table', 'text', 'mixed'] indicating the source type of the evidence
evidence_count: [int]          # The total count of all evidence related to the question
distractor_count: [int]        # The total number of distractor items, meaning information blocks that are irrelevant or misleading for the answer
info_count: [int]              # The total number of information blocks in the document, including text, tables, images, etc.
text_2_idx: [dict[str, str]]   # A dictionary mapping text information to corresponding indices
idx_2_text: [dict[str, str]]   # A reverse dictionary mapping indices back to the corresponding text content
image_2_idx: [dict[str, str]]  # A dictionary mapping image paths to corresponding indices
idx_2_image: [dict[str, str]]  # A reverse dictionary mapping indices back to image paths
table_2_idx: [dict[str, str]]  # A dictionary mapping table paths to corresponding indices
idx_2_table: [dict[str, str]]  # A reverse dictionary mapping indices back to table paths
meta_data: [dict]              # Additional metadata used during the construction of the data
distractor_contents: [list]    # Similar to `evidence_contents`, but contains distractors, which are irrelevant or misleading information
```