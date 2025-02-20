**Note:**  
Before running the scripts, make sure to follow the instructions in `MCiteBench/data/README.md` and verify that your data(example or full dataset) and API key are properly set up.

## Supported Models and Usage

This repository provides scripts that support the following models:
- gpt-4o-2024-11-20 / gpt-4o-mini
- Qwen2-VL-7B-Instruct / Qwen2-VL-72B-Instruct
- InternVL2_5-8B / InternVL2_5-78B
- MiniCPM-V-2_6
- llava-onevision-qwen2-7b-ov-hf / llava-onevision-qwen2-7b-ov-chat-hf
- Llama-3.2-11B-Vision-Instruct / Llama-3.2-90B-Vision-Instruct

## Running Scripts for Different Models

Each model has its own script. Below are the commands to run the respective scripts:

1. gpt-4o-2024-11-20 / gpt-4o-mini
```bash
cd generation
# gpt-4o-2024-11-20 or gpt-4o-mini
python run_api_based.py --model gpt-4o-2024-11-20
```
2. Qwen2-VL-7B-Instruct / Qwen2-VL-72B-Instruct / InternVL2_5-8B / InternVL2_5-78B / MiniCPM-V-2_6
```bash
cd generation
# Specify model name or your local path
python run_vllm_based.py --model Qwen2-VL-7B-Instruct --gpu_num 1 --batch_size 32
```
3. llava-onevision-qwen2-7b-ov-hf / llava-onevision-qwen2-7b-ov-chat-hf
```bash
cd generation
# Specify model name or your local path
CUDA_VISIBLE_DEVICES=0 python run_llava.py --model llava-onevision-qwen2-7b-ov-hf
# Works on 1 GPU
```
4. Llama-3.2-11B-Vision-Instruct / Llama-3.2-90B-Vision-Instruct
```bash
cd generation
# Specify model name or your local path
python run_mllama.py --model Llama-3.2-11B-Vision-Instruct
```

After running the scripts, the corresponding result files will be generated in the ../output/ directory.
