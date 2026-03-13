#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_iagent.sh [dataset] [domain] [agent_type] [model_name]
#
# Example (Qwen on DashScope):
#   export DASHSCOPE_API_KEY="your_key"
#   bash run_iagent.sh amazon books static qwen-plus

DATASET="${1:-amazon}"
DOMAIN="${2:-books}"
AGENT_TYPE="${3:-static}"   # static | dynamic
MODEL_NAME="${4:-qwen-plus}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}"

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

