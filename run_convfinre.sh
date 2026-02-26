#!/bin/bash

export DEEPSEEK_API_KEY='DeepSeek API KEY'

#personal
export OPENAI_API_KEY='OPENAI API KEY'
export TOGETHER_AI_API_KEY='Togetherai API KEY'

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_TOKEN='YOUR HF TOKEN'
export HF_HUB_READ_TIMEOUT=180
export HF_HUB_CONNECT_TIMEOUT=180

set -e

SHOTS=(0)

MODELS=(
    "gpt-4o"
    "gpt-5.2"
)

OUT_DIR="results/conv_finre_nohis"

for MODEL in "${MODELS[@]}"; do
  echo "======================================="
  echo "running model: $MODEL"
  echo "======================================="

  for SHOT in "${SHOTS[@]}"; do
    lm_eval --model openai-chat-completions \
      --model_args "pretrained=$MODEL" \
      --tasks ConvFinRe \
      --num_fewshot "$SHOT" \
      --use_cache "./cache_$MODEL" \
      --batch_size auto \
      --output_path "${OUT_DIR}" \
      --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-convfinre-nohis-results,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
      --log_samples \
      --apply_chat_template \
      --include_path ./tasks/ConvFinRe

    sleep 2
  done

  sleep 3
done


echo "Evaluation completed successfully!"
