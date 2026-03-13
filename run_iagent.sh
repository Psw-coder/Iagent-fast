#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_iagent.sh [dataset] [domain] [agent_type] [model_name]
#
# Example (Qwen on DashScope):
#   export DASHSCOPE_API_KEY="sk-9fb75c4879bf402492463d0c0bb11bc0"   deepseek
#    export DASHSCOPE_API_KEY="sk-0550bea09f284b2ba455652095e4f25f"   qwen
#   bash run_iagent.sh amazon books static qwen-plus


# openai  model name gpt-4o-mini    url   https://yunwu.ai/v1/chat/completions   key    export DASHSCOPE_API_KEY="sk-8hOLy1a2N8kKyA3YjMeq4bxYr4wnF5LhA4Pv7oxgwwbEc0fh"
DATASET="${1:-amazon}"
DOMAIN="${2:-books}"
AGENT_TYPE="${3:-static}"   # static | dynamic
MODEL_NAME="${4:-gpt-4o-mini}"
BASE_URL="${BASE_URL:-https://yunwu.ai/v1}"
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}"




# DATASET="${1:-amazon}"
# DOMAIN="${2:-books}"
# AGENT_TYPE="${3:-static}"   # static | dynamic
# MODEL_NAME="${4:-deepseek-reasoner}"
# BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"
# API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}"

if [[ -z "${API_KEY}" ]]; then
  echo "Error: API key is empty. Set one of: API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY"
  exit 1
fi

echo "Run config:"
echo "  dataset=${DATASET}"
echo "  domain=${DOMAIN}"
echo "  agent_type=${AGENT_TYPE}"
echo "  model_name=${MODEL_NAME}"
echo "  base_url=${BASE_URL}"

python main_iagent_mp.py \
  --dataset "${DATASET}" \
  --domain "${DOMAIN}" \
  --agent_type "${AGENT_TYPE}" \
  --model_name "${MODEL_NAME}" \
  --base_url "${BASE_URL}" \
  --api_key "${API_KEY}"

