# 模块D并行提速报告

## 1. 报告目的

- 目标：验证模块D从“单进程多输出批渲染”切换到“受控并行多进程单段渲染”后，是否在同一音频输入上取得稳定提速。
- 范围：仅比较模块D耗时（`render_segments_elapsed`、`total_elapsed`）与 `gpu_to_cpu_fallback_count`，不改变模块A/B/C产物。

## 2. 实现变更与代码证据

### 2.1 配置层

- 新增并启用受控并发字段：`render_workers`（默认 `3`）。
  - 证据：[src/music_video_pipeline/config.py:84](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/config.py:84)
- `render_batch_size` 降为兼容字段，默认 `1`。
  - 证据：[src/music_video_pipeline/config.py:83](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/config.py:83)

### 2.2 渲染执行层（模块D）

- `render_batch_size` 在新路径固定按 `1` 执行；若配置不为 `1` 仅告警。
  - 证据：[src/music_video_pipeline/modules/module_d.py:96](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:96)
- `render_workers` 归一化范围 `1~4`，非法回退 `3`。
  - 证据：[src/music_video_pipeline/modules/module_d.py:123](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:123)
- 单段命令语义：每段固定“单输入单输出”。
  - 证据：[src/music_video_pipeline/modules/module_d.py:264](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:264)
- 原子写入：先写 `segment_xxx.tmp.mp4`，成功后 `replace` 成 `segment_xxx.mp4`；失败清理临时文件。
  - 证据：[src/music_video_pipeline/modules/module_d.py:305](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:305)
- 并发模型：`ProcessPoolExecutor` + `spawn`，避免同进程内多路编码互相影响。
  - 证据：[src/music_video_pipeline/modules/module_d.py:599](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:599)
- 单段失败隔离：GPU失败仅该段CPU重试一次，不重跑已成功段。
  - 证据：[src/music_video_pipeline/modules/module_d.py:644](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:644)
- 顺序稳定：并发完成可乱序，但输出按 `segment_index` 重新排序。
  - 证据：[src/music_video_pipeline/modules/module_d.py:678](/home/sod2204/work/zonghe/t1/src/music_video_pipeline/modules/module_d.py:678)

## 3. 测试设计

### 3.1 测试对象

- 同一任务：`wuli_ab_compare_20260406_171137`
- 同一输入音频：`resources/wuli.m4a`（任务内复用，不重跑A/B/C）
- 统一方式：`--force-module D` 仅重跑模块D

### 3.2 指标口径

- `render_segments_elapsed`：模块D渲染阶段耗时。
- `total_elapsed`：模块D总耗时（渲染+拼接）。
- `gpu_to_cpu_fallback_count`：GPU失败后单段CPU重试成功计数。

## 4. 运行结果

### 4.1 历史三组对比（workers=1/2/3）

| workers | render_segments_elapsed(s) | total_elapsed(s) | gpu_to_cpu_fallback_count | 日志 |
|---|---:|---:|---:|---|
| 1 | 54.379 | 57.473 | 0 | [run_20260406_182122_315354.log](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_182122_315354.log:97) |
| 2 | 28.787 | 31.547 | 0 | [run_20260406_182220_357533.log](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_182220_357533.log:97) |
| 3 | 22.787 | 25.662 | 0 | [run_20260406_182252_483481.log](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_182252_483481.log:97) |

### 4.2 本轮追加对比（workers=3 vs 4）

| workers | render_segments_elapsed(s) | total_elapsed(s) | gpu_to_cpu_fallback_count | hard_fail_count | 日志 |
|---|---:|---:|---:|---:|---|
| 3 | 22.948 | 25.786 | 0 | 0 | [run_20260406_192457_148245.log](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_192457_148245.log:97) |
| 4 | 19.541 | 22.438 | 0 | 0 | [run_20260406_192523_193015.log](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_192523_193015.log:97) |

对应耗时统计证据：
- workers=3：[run_20260406_192457_148245.log:98](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_192457_148245.log:98)
- workers=4：[run_20260406_192523_193015.log:98](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_192523_193015.log:98)

### 4.3 提速结论（本轮3 vs 4）

- workers=4 相对 workers=3：
  - `render_segments_elapsed` 提速 `1.174x`（下降 `14.85%`）
  - `total_elapsed` 提速 `1.149x`（下降 `12.98%`）
- 两组都未触发 GPU->CPU 回退（`gpu_to_cpu_fallback_count=0`）。

## 5. 历史问题对照（改造前证据）

- 旧“单命令多输出批渲染（batch_size=12）”曾出现 NVENC 会话错误并导致性能退化：
  - `OpenEncodeSessionEx failed`、`No capable devices found`。
  - 证据：[run_20260406_171710_549362.log:138](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_171710_549362.log:138)
- 当时耗时：`render_segments_elapsed=82.490s`、`total_elapsed=85.236s`。
  - 证据：[run_20260406_171710_549362.log:943](/mnt/c/Users/QWERT/Desktop/runs/wuli_ab_compare_20260406_171137/log/run_20260406_171710_549362.log:943)

## 6. 结论与建议

1. 本机（RTX 4060 Laptop）在当前任务规模（86段）下，受控并行方案有效，`workers=4` 在本轮单次测试中最快。
2. 配置默认值建议保持 `render_workers=3`（稳定性优先）。
3. 若你在同机同场景下追求极限速度，可临时升到 `render_workers=4`；建议保留 `gpu_to_cpu_fallback_count` 监控并至少重复跑3次取中位数后再固定。
4. 建议持续监控三项核心指标：`render_segments_elapsed`、`total_elapsed`、`gpu_to_cpu_fallback_count`。

## 7. 附：本轮输出记录

- 历史三组汇总：`/tmp/module_d_workers_compare_20260406_182122.tsv`
- 本轮3v4汇总：`/tmp/module_d_workers_3v4_20260406_192457.tsv`
