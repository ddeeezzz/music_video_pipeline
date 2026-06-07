"""
benchmark_dsv4flash_temps.py
用途：测硅基流动 DeepSeek-V4-Flash 在温度 0.5/0.7/0.9 下的速度。
用法：python tests/benchmark_dsv4flash_temps.py
"""
import json, logging, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from music_video_pipeline.config import ModuleBLlmConfig
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S.%f")
logger = logging.getLogger("bench_v4flash")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_KEY = (PROJECT_ROOT / ".secrets" / "siliconflow_api_key.txt").read_text(encoding="utf-8").strip()

user_template = (
    "## 故事\n"
    "黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。\n\n"
    "## 意象\n"
    "少女：水手服少女，黑长直，细瘦身形。\n"
    "黑猫：瘦长、警觉的黑猫，细尾，尖耳。"
)

temperatures = [0.5, 0.7, 0.9]
results = []

for t in temperatures:
    config = ModuleBLlmConfig(
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V4-Flash",
        api_key_file=".secrets/siliconflow_api_key.txt",
        timeout_seconds=120.0,
        retry_times=0,
        temperature=t,
        top_p=0.9,
        stream=True,
    )

    describer = Role1ImageryDescriber(logger=logger, llm_config=config, project_root=PROJECT_ROOT)

    print(f"\n{'='*56}")
    print(f"  温度: {t}")
    print(f"{'='*56}")

    t0 = time.perf_counter()
    try:
        result = describer.generate(user_template)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"OK | 意象数: {len(result)} | 耗时: {elapsed:.1f}ms ({elapsed/1000:.2f}s)")
        for item in result:
            print(f"  - {item.imagery_name}: {str(item.pos_zh)[:60]}...")
        results.append({"temperature": t, "success": True, "elapsed_ms": round(elapsed, 1), "imagery_count": len(result)})
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"FAIL | 耗时: {elapsed:.1f}ms | 错误: {e}")
        results.append({"temperature": t, "success": False, "elapsed_ms": round(elapsed, 1), "error": str(e)})

print("\n")
print("=" * 56)
print("  📊 DeepSeek-V4-Flash 温度测速汇总")
print("=" * 56)
print(f"  {'温度':<8} {'耗时(ms)':<12} {'耗时(s)':<10} {'意象数':<8} {'状态'}")
print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*6}")
for r in results:
    status = "✅" if r["success"] else "❌"
    t = r["temperature"]
    ms = r["elapsed_ms"]
    s = f"{ms/1000:.2f}" if ms else "N/A"
    cnt = r.get("imagery_count", "-")
    print(f"  {t:<8} {ms:<12} {s:<10} {cnt:<8} {status}")
print("=" * 56)
