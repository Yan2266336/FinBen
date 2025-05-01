#!/bin/bash

# openai 
export OPENAI_API_KEY='your openai api key'
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_TOKEN='your hf token'
set -e

MODELS=(
    "google/gemma-3-27b-it"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-Omni-7B"
    "Duxiaoman-DI/Llama3.1-XuanYuan-FinX1-Preview
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    "TheFinAI/plutus-8B-instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    )


# Run the Hugging Face VLLM evaluation command
for MODEL in "${MODELS[@]}"; do
    echo "running model: $MODEL"
    # Run models
    lm_eval --model vllm \
            --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.85,max_model_len=8192" \
            --tasks en \
            --batch_size auto \
            --num_fewshot 0 \
            --output_path results/english \
            --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-multifin-en,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
            --log_samples \
            --apply_chat_template \
            --include_path ./tasks
    sleep 2
done


# output message
echo "Evaluation completed successfully!"
