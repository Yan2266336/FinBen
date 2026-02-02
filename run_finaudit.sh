#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_USE_V1=0
export HF_TOKEN='your hf token'
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_CONNECT_TIMEOUT=180
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


set -e



MODELS=(
    # "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen3-32B"
    # "google/gemma-3-27b-it"
    # "google/gemma-3-12b-it"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "TheFinAI/Fin-o1-14B"
    # "SUFE-AIFLM-Lab/Fin-R1"
    )

TASKS=(
    "FinSMgen"
    "FinREgen"
    "FinMRgen"
    )


##Run the Hugging Face VLLM evaluation command
for MODEL in "${MODELS[@]}"; do
    echo "running model: $MODEL"
    for TASK in "${TASKS[@]}"; do
        echo "runing task: $TASK"
        lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.80,max_model_len=81920,rope_scaling.rope_type=yarn,rope_scaling.factor=2.5,rope_scaling.original_max_position_embeddings=32768" \
        --tasks "$TASK" \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path results/finauditing \
        --hf_hub_log_args "hub_results_org=your_org_name,details_repo_name=your_repo_name,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks/FinAuditing
        
        sleep 2
    done
        
    sleep 3
done



        
# output message
echo "Evaluation completed successfully!"
