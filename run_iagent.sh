#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Preset mode (recommended):
#     bash run_iagent.sh <1|2|3|4|5|6|7|8|all> [model_name]
#
#   Manual mode (backward-compatible):
#     bash run_iagent.sh <dataset> <domain> <agent_type> [model_name]
#
# Presets (from README):
#   1: amazon books   static
#   2: amazon movietv static
#   3: good   reads   static
#   4: yelp   yelp    static
#   5: amazon books   dynamic
#   6: amazon movietv dynamic
#   7: good   reads   dynamic
#   8: yelp   yelp    dynamic
#
# Example:
#   export OPENAI_API_KEY="<your_key>"
#   bash run_iagent.sh 1
#   bash run_iagent.sh all qwen-plus

BASE_URL="${BASE_URL:-https://yunwu.ai/v1}"
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}"
DEFAULT_MODEL="${DEFAULT_MODEL:-gpt-4o-mini}"

if [[ -z "${API_KEY}" ]]; then
  echo "Error: API key is empty. Set one of: API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY"
  exit 1
fi

print_help() {
  echo "Usage:"
  echo "  Preset mode: bash run_iagent.sh <1|2|3|4|5|6|7|8|all> [model_name]"
  echo "  Manual mode: bash run_iagent.sh <dataset> <domain> <agent_type> [model_name]"
  echo
  echo "Examples:"
  echo "  bash run_iagent.sh 1"
  echo "  bash run_iagent.sh all qwen-plus"
  echo "  bash run_iagent.sh amazon books static gpt-4o-mini"
}

run_one() {
  local dataset="$1"
  local domain="$2"
  local agent_type="$3"
  local model_name="$4"

  echo "Run config:"
  echo "  dataset=${dataset}"
  echo "  domain=${domain}"
  echo "  agent_type=${agent_type}"
  echo "  model_name=${model_name}"
  echo "  base_url=${BASE_URL}"
  echo

  python main_iagent_mp.py \
    --dataset "${dataset}" \
    --domain "${domain}" \
    --agent_type "${agent_type}" \
    --model_name "${model_name}" \
    --base_url "${BASE_URL}" \
    --api_key "${API_KEY}"
}

run_preset() {
  local id="$1"
  local model_name="$2"

  case "${id}" in
    1) run_one "amazon" "books" "static" "${model_name}" ;;
    2) run_one "amazon" "movietv" "static" "${model_name}" ;;
    3) run_one "good"   "reads" "static" "${model_name}" ;;
    4) run_one "yelp"   "yelp"  "static" "${model_name}" ;;
    5) run_one "amazon" "books" "dynamic" "${model_name}" ;;
    6) run_one "amazon" "movietv" "dynamic" "${model_name}" ;;
    7) run_one "good"   "reads" "dynamic" "${model_name}" ;;
    8) run_one "yelp"   "yelp"  "dynamic" "${model_name}" ;;
    *)
      echo "Error: invalid preset id '${id}'. Use 1-8 or all."
      exit 1
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  print_help
  exit 1
fi

FIRST_ARG="$1"

if [[ "${FIRST_ARG}" == "all" ]]; then
  MODEL_NAME="${2:-${DEFAULT_MODEL}}"
  for id in 1 2 3 4 5 6 7 8; do
    echo "========== Running preset ${id} =========="
    run_preset "${id}" "${MODEL_NAME}"
  done
elif [[ "${FIRST_ARG}" =~ ^[1-8]$ ]]; then
  MODEL_NAME="${2:-${DEFAULT_MODEL}}"
  run_preset "${FIRST_ARG}" "${MODEL_NAME}"
else
  DATASET="$1"
  DOMAIN="${2:-books}"
  AGENT_TYPE="${3:-static}"
  MODEL_NAME="${4:-${DEFAULT_MODEL}}"
  run_one "${DATASET}" "${DOMAIN}" "${AGENT_TYPE}" "${MODEL_NAME}"
fi

