set -euo pipefail      
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
LOGFILE="${SCRIPT_DIR}/logs/auto_$(date +'%Y%m%d_%H%M%S').log"
echo "========  Auto pipeline started at $(date)  ========" | tee -a "$LOGFILE"
run_step () {
  local name="$1"; shift
  echo -e "\n>>> [$name] $(date)" | tee -a "$LOGFILE"
  # 用 tee 实时把 stdout + stderr 写入日志
  "$@" 2>&1 | tee -a "$LOGFILE"
}
# 1) 生成对抗样本、污染语料
run_step "Attack_hotpotqa" python Attack.py \
  --dataset      hotpotqa \
  --data_path    ./datasets/hotpotqa \
  --out_root     ./poison \
  --model_name   /root/autodl-tmp/MODEL/Llama-2-7b-chat-hf \
  --attack_method LM_targeted \
  --adv_per_query 5

run_step "Attack_nq" python Attack.py \
  --dataset      nq \
  --data_path    ./datasets/nq \
  --out_root     ./poison \
  --model_name   /root/autodl-tmp/MODEL/Llama-2-7b-chat-hf \
  --attack_method LM_targeted \
  --adv_per_query 5

# 2) 构建 / 更新向量数据库（clean & poisoned）
run_step "BuildStorage_nq"  python BuildStorage.py \
  --data_path   ./poison/nq \
  --model_path  /root/autodl-tmp/MODEL/contriever \
  --output_dir  ./Storage/Poison/contriever/nq \
  --batch_size  256

run_step "BuildStorage_hotpot"  python BuildStorage.py \
  --data_path   ./poison/hotpotqa \
  --model_path  /root/autodl-tmp/MODEL/contriever \
  --output_dir  ./Storage/Poison/contriever/hotpotqa \
  --batch_size  256

# 3) 检索并生成答案，输出比较 JSON
#run_step "SearchRAG.py"    python SearchRAG.py
echo -e "\n========  Pipeline finished at $(date)  ========" | tee -a "$LOGFILE"
echo "日志已保存：$LOGFILE"
