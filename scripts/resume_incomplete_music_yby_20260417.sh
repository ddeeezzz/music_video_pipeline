#!/usr/bin/env bash
set -u
cd /root/data/t1

echo "[resume-batch] start $(date '+%F %T %z')"

run_one() {
  local task_id="$1"
  local config_path="$2"
  echo "[resume-batch] ===== ${task_id} resume start $(date '+%F %T %z') ====="
  ./.venv/bin/mvpl resume --task-id "$task_id" --config "$config_path"
  local rc=$?
  echo "[resume-batch] ===== ${task_id} resume end rc=${rc} $(date '+%F %T %z') ====="
  return $rc
}

FAIL=0
run_one "jieranduhuo01" "/root/data/t1/configs/music_yby/jieranduhuo_v2.json" || FAIL=1
run_one "juebieshu01" "/root/data/t1/configs/music_yby/juebieshu_v2.json" || FAIL=1
run_one "tots01" "/root/data/t1/configs/music_yby/tots_v2.json" || FAIL=1

echo "[resume-batch] finish FAIL=${FAIL} $(date '+%F %T %z')"
exit ${FAIL}
