#!/usr/bin/env bash
# Every tokenizer on every model, three rounds, each measurement a fresh process on one core.
# The tokenizers alternate within a round, so a slow drift of the machine's clock touches them alike.
# HF tokenizers 0.23 and 1.0 cannot share an environment: each interpreter is named here.
#
#   TOKNROLL_PY=python3 HF1_PY=venv-hf1/bin/python HF023_PY=venv-hf023/bin/python ./run.sh > results.jsonl
set -euo pipefail
cd "$(dirname "$0")"
CORE=${CORE:-3}
MODELS=(Qwen/Qwen3-0.6B unsloth/Llama-3.2-1B-Instruct openai/gpt-oss-20b deepseek-ai/DeepSeek-V3.2
        google/gemma-4-e2b-it mistralai/Mistral-7B-Instruct-v0.3)
for round in 1 2 3; do
  for model in "${MODELS[@]}"; do
    taskset -c "$CORE" "$TOKNROLL_PY" encode.py toknroll "$model"
    taskset -c "$CORE" "$HF1_PY" encode.py tokenizers "$model"
    taskset -c "$CORE" "$HF023_PY" encode.py tokenizers "$model"
    if [ "$model" = openai/gpt-oss-20b ]; then
      taskset -c "$CORE" "$HF023_PY" encode.py tiktoken "$model" --encoding o200k_harmony
    fi
  done
done
