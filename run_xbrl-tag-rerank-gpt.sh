#!/bin/bash

export DEEPSEEK_API_KEY=''

export OPENAI_API_KEY=''

export HF_TOKEN=''

set -e


# Run the Hugging Face VLLM evaluation command

lm_eval --model openai-chat-completions \
    --model_args "model=gpt-4o" \
    --tasks GPT4o_NEN \
    --num_fewshot 0 \
    --output_path results/fincl_rerank \
    --use_cache "./cache_gpt4o" \
    --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-fintag-rerank-pipeline-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
    --log_samples \
    --apply_chat_template \
    --include_path ./tasks

# output message
echo "Evaluation completed successfully!"
