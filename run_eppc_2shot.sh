#!/bin/bash

export HF_TOKEN='your hf token'
set -e

## here it means the running rounds not the shots
SHOTS=(
        0
        1
        2
        3
        4
      )

## replace with your models
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "meta-llama/Llama-3.1-70B-Instruct"
    )


# Run the Hugging Face VLLM evaluation command
for MODEL in "${MODELS[@]}"; do
    echo "running model: $MODEL"
    for SHOT in "${SHOTS[@]}"; do
        lm_eval --model vllm \
                 --model_args "pretrained=$MODEL,tensor_parallel_size=4,gpu_memory_utilization=0.90,max_model_len=24676" \
                 --tasks EppcExtraction \
                 --num_fewshot 2 \
                 --batch_size auto \
                 --output_path results/eppc \
                 --hf_hub_log_args "hub_results_org=YanAdjeNole,details_repo_name=eppc-2shot,push_results_to_hub=True,push_samples_to_hub=True,public_repo=True" \
                 --log_samples \
                 --apply_chat_template \
                 --include_path ./tasks
        sleep 1
    done
    sleep 3
done
        
# output message
echo "Evaluation completed successfully!"
