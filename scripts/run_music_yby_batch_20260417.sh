#!/usr/bin/env bash
set -euo pipefail
cd /root/data/t1
echo "[batch] start $(date '+%F %T %z')"
echo "[batch] run jieranduhuo01"
./.venv/bin/mvpl run --task-id jieranduhuo01 --config /root/data/t1/configs/music_yby/jieranduhuo_v2.json
echo "[batch] run juebieshu01"
./.venv/bin/mvpl run --task-id juebieshu01 --config /root/data/t1/configs/music_yby/juebieshu_v2.json
echo "[batch] run tots01"
./.venv/bin/mvpl run --task-id tots01 --config /root/data/t1/configs/music_yby/tots_v2.json
echo "[batch] done $(date '+%F %T %z')"
