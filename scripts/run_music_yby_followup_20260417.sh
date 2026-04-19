#!/usr/bin/env bash
set -euo pipefail
cd /root/data/t1

wait_task_terminal() {
  local task_id="$1"
  while true; do
    local status
    status=$(./.venv/bin/python - <<PY
import sqlite3
con=sqlite3.connect('/root/data/runs/pipeline_state.sqlite3')
cur=con.cursor()
row=cur.execute("SELECT status FROM tasks WHERE task_id=?", ('${task_id}',)).fetchone()
print(row[0] if row else '')
con.close()
PY
)
    if [[ "$status" == "done" || "$status" == "failed" ]]; then
      echo "[followup] task=${task_id} terminal status=${status} at $(date '+%F %T %z')"
      break
    fi
    echo "[followup] waiting task=${task_id}, current_status=${status:-<missing>} at $(date '+%F %T %z')"
    sleep 20
  done
}

echo "[followup] start $(date '+%F %T %z')"
wait_task_terminal "jieranduhuo01"

echo "[followup] run juebieshu01"
./.venv/bin/mvpl run --task-id juebieshu01 --config /root/data/t1/configs/music_yby/juebieshu_v2.json

echo "[followup] run tots01"
./.venv/bin/mvpl run --task-id tots01 --config /root/data/t1/configs/music_yby/tots_v2.json

echo "[followup] done $(date '+%F %T %z')"
