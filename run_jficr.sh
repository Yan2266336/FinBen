#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_USE_V1=0
export HF_TOKEN='hf-token'
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_CONNECT_TIMEOUT=180
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


set -e



MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    )


##Run the Hugging Face VLLM evaluation command
for MODEL in "${MODELS[@]}"; do
    echo "running model: $MODEL"
    lm_eval --model vllm \
    --model_args "pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.80,max_model_len=1024" \
    --tasks JFICR \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path results/JP \
    --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-japanese-ICR-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    --log_samples \
    --apply_chat_template \
    --include_path ./tasks/japanese
        
    sleep 3
done



        
# output message
echo "Evaluation completed successfully!"
