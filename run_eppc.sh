#!/bin/bash

export OPENAI_API_KEY=''   
export DEEPSEEK_API_KEY=''
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_TOKEN=''
set -e


MODELS=(
    "hongzhouyu/FineMedLM-o1"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # "aaditya/Llama3-OpenBioLLM-70B"
    # "YanAdjeNole/sdoh-llama-3.3-70b"
    )


# Run the Hugging Face VLLM evaluation command
for MODEL in "${MODELS[@]}"; do
    echo "running model: $MODEL"
    lm_eval --model vllm \
            --model_args "pretrained=$MODEL,tensor_parallel_size=4,dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=8192" \
            --tasks EppcExtraction \
            --num_fewshot 0 \
            --batch_size auto \
            --output_path results/eppc \
            --hf_hub_log_args "hub_results_org=YanAdjeNole,details_repo_name=lm-eval-ranking-0-shot-results-new,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
            --log_samples \
            --apply_chat_template \
            --include_path ./tasks/eppc
    sleep 3
done

echo "Evaluation completed successfully!"
