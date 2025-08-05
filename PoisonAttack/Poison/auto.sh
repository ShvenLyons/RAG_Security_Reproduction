#!/usr/bin/env bash
set -euo pipefail

# === 基础设置 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
LOGFILE="${SCRIPT_DIR}/logs/autoPoison_$(date +'%Y%m%d_%H%M%S').log"

echo "========  Auto-Poison pipeline started at $(date)  ========" | tee -a "$LOGFILE"

run_step () {
  local name="$1"; shift
  echo -e "\n>>> [$name] $(date)" | tee -a "$LOGFILE"
  # 实时把 stdout + stderr 写入日志
  "$@" 2>&1 | tee -a "$LOGFILE"
}

# === 统一参数 ===
MODEL_LLM="/root/autodl-tmp/MODEL/Llama-2-7b-chat-hf"
MODEL_EMB="/root/autodl-tmp/MODEL/contriever"
POISON_ROOT="./poison"
CLEAN_ROOT="./Storage/contriever"
POISON_STORE_ROOT="./Storage/Poison/contriever"
ADV_PER_Q=5
BATCH_SIZE=256
TOP_K=5
MAX_TOKENS=32

# === 依次处理 3 个数据集 ===
DATASETS=("hotpotqa" "nq" "msmarco")

for DS in "${DATASETS[@]}"; do
  ############ 1) 生成对抗样本、污染语料 ################
  run_step "Attack_${DS}" \
    python Attack.py \
      --dataset        "${DS}" \
      --data_path      "./datasets/${DS}" \
      --out_root       "${POISON_ROOT}" \
      --model_name     "${MODEL_LLM}" \
      --attack_method  "LM_targeted" \
      --adv_per_query  "${ADV_PER_Q}"

  ############ 2) 构建 / 更新向量数据库 ##################
  run_step "BuildStorage_${DS}" \
    python BuildStorage.py \
      --data_path   "${POISON_ROOT}/${DS}" \
      --model_path  "${MODEL_EMB}" \
      --output_dir  "${POISON_STORE_ROOT}/${DS}" \
      --batch_size  "${BATCH_SIZE}"

  ############ 3) RAG 检索 & 生成，对比输出 ##############
  run_step "RAGcompare_${DS}" \
    python RAGcompare.py \
      --dataset      "${DS}" \
      --clean_root   "${CLEAN_ROOT}" \
      --poison_root  "${POISON_STORE_ROOT}" \
      --model_emb    "${MODEL_EMB}" \
      --model_llm    "${MODEL_LLM}" \
      --top_k        "${TOP_K}" \
      --max_tokens   "${MAX_TOKENS}" \
      --output       "${SCRIPT_DIR}/validation_${DS}.json"
done

echo -e "\n========  Pipeline finished at $(date)  ========" | tee -a "$LOGFILE"
echo "日志已保存：$LOGFILE"
