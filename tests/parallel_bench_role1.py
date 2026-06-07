"""
parallel_bench_role1.py
用途：并发测速 role1 — 3个模型 x 3个温度，9个请求同时发出。
通过 ThreadPoolExecutor 实现真正的并发 HTTP 请求。
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from music_video_pipeline.config import ModuleBLlmConfig
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parallel_bench")


@dataclass
class BenchResult:
    label: str
    model: str
    temperature: float
    success: bool
    elapsed_ms: float
    imagery_count: int = 0
    error: str = ""
    items: list = field(default_factory=list)


def _bench_one(label: str, llm_config: ModuleBLlmConfig, user_md: str, project_root: Path) -> BenchResult:
    """同步函数：跑一次 role1 端到端。"""
    describer = Role1ImageryDescriber(
        logger=logger,
        llm_config=llm_config,
        project_root=project_root,
    )

    t0 = time.perf_counter()
    try:
        result = describer.generate(user_md)
        elapsed = (time.perf_counter() - t0) * 1000
        items = [(r.imagery_name, r.pos_zh[:80]) for r in result]
        return BenchResult(
            label=label,
            model=llm_config.model if isinstance(llm_config.model, str) else str(llm_config.model),
            temperature=llm_config.temperature,
            success=True,
            elapsed_ms=round(elapsed, 1),
            imagery_count=len(result),
            items=items,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return BenchResult(
            label=label,
            model=llm_config.model if isinstance(llm_config.model, str) else str(llm_config.model),
            temperature=llm_config.temperature,
            success=False,
            elapsed_ms=round(elapsed, 1),
            error=str(e)[:300],
        )


def main():
    project_root = PROJECT_ROOT

    # 加载基础配置
    cfg_file = project_root / "configs" / "music_windows_4060" / "jieranduhuo.json"
    if not cfg_file.exists():
        print(f"[ERROR] 配置文件不存在: {cfg_file}")
        return
    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    llm_raw = raw["module_b"]["llm"]

    base_config = {
        "provider": str(llm_raw.get("provider", "siliconflow")).strip(),
        "base_url": str(llm_raw.get("base_url", "https://api.siliconflow.cn/v1")).strip(),
        "api_key_file": str(llm_raw.get("api_key_file", ".secrets/siliconflow_api_key.txt")).strip(),
        "timeout_seconds": float(llm_raw.get("timeout_seconds", 120.0)),
        "first_chunk_timeout_seconds": float(llm_raw.get("first_chunk_timeout_seconds", 10.0)),
        "retry_times": int(llm_raw.get("retry_times", 1)),
        "top_p": float(llm_raw.get("top_p", 0.9)),
        "stream": True,
    }

    models = [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "deepseek-ai/DeepSeek-V3",
    ]
    temperatures = [0.3, 0.5, 0.7]

    configs = []
    for model in models:
        for temp in temperatures:
            cfg = ModuleBLlmConfig(**base_config, model=model, temperature=temp)
            short_model = model.split("/")[-1]
            label = f"{short_model} (t={temp})"
            configs.append((label, cfg))

    user_md = (
        "## 故事\n"
        "黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。\n\n"
        "## 意象\n"
        "少女：水手服少女，黑长直，细瘦身形。\n"
        "黑猫：瘦长、警觉的黑猫，细尾，尖耳。"
    )

    print()
    print("=" * 66)
    print("  Role1 并行测速：3模型 x 3温度 = 9请求同时发出")
    print("=" * 66)
    for label, cfg in configs:
        print(f"    {label:>30}  |  stream={cfg.stream}")
    print()

    print(f"[{time.strftime('%H:%M:%S')}] 开始并发请求...\n")
    t_all_start = time.perf_counter()

    # ThreadPoolExecutor 实现真正并发
    results: list[BenchResult] = []
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_map = {
            executor.submit(_bench_one, label, cfg, user_md, project_root): label
            for label, cfg in configs
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result.success else "FAIL"
                print(f"  [{time.strftime('%H:%M:%S')}] [{status}] {label:>30}  {result.elapsed_ms:>8.1f} ms")
            except Exception as e:
                print(f"  [{time.strftime('%H:%M:%S')}] [CRASH] {label:>30}  {e}")

    t_all_end = time.perf_counter()
    wall_clock = (t_all_end - t_all_start) * 1000
    print(f"\n[{time.strftime('%H:%M:%S')}] 全部完成，墙上总耗时: {wall_clock:.0f} ms ({wall_clock/1000:.1f} s)")

    # ----- 按模型分组输出 -----
    model_order = ["Qwen2.5-32B-Instruct", "Qwen2.5-14B-Instruct", "DeepSeek-V3"]
    print(f"\n{'='*66}")
    print("  各模型/温度详情")
    print(f"{'='*66}")
    for model_name in model_order:
        group = sorted([r for r in results if r.model and model_name in r.model], key=lambda r: r.temperature)
        if not group:
            continue
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n  ── {short} ──")
        print(f"  {'temp':>6} | {'耗时':>8} | {'意象数':>5} | 内容预览")
        print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*5}-+-{'─'*30}")
        for r in group:
            if r.success:
                preview = r.items[0][1][:55] if r.items else "(空)"
                print(f"  t={r.temperature:<4.1f} | {r.elapsed_ms:>7.1f}ms | {r.imagery_count:>3}个 | {preview}")
            else:
                print(f"  t={r.temperature:<4.1f} | {'FAIL':>8} | {'':>5} | {r.error[:55]}")

    # ----- 速度排行 -----
    print(f"\n{'='*66}")
    print("  ⚡ 速度排行 (快→慢)")
    print(f"{'='*66}")
    sorted_results = sorted([r for r in results if r.success], key=lambda r: r.elapsed_ms)
    for i, r in enumerate(sorted_results, 1):
        short_model = r.model.split("/")[-1] if "/" in r.model else r.model
        print(f"  #{i:>2}  {r.elapsed_ms:>8.1f} ms  | {short_model:>25}  t={r.temperature}")

    # ----- 质量对比：同一模型不同温度下第一个意象的描述 -----
    print(f"\n{'='*66}")
    print("  🔍 质量对比 (第一个意象的描述)")
    print(f"{'='*66}")
    for model_name in model_order:
        group = sorted([r for r in results if r.model and model_name in r.model and r.success],
                       key=lambda r: r.temperature)
        if len(group) < 2:
            continue
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n  ── {short} ──")
        for r in group:
            print(f"  t={r.temperature}  ({r.elapsed_ms:.0f}ms)")
            for name, desc in r.items:
                print(f"    {name}: {desc}")
            if len(r.items) < 2:
                print(f"    (只有 {len(r.items)} 个意象)")

    print(f"\n{'='*66}")
    print("  测速完成")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
