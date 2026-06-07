"""补测 V4-Flash 温度0.7"""
import json, logging, sys, time, os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from music_video_pipeline.config import ModuleBLlmConfig
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber

user_template = (
    "## 故事\n"
    "黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。\n\n"
    "## 意象\n"
    "少女：水手服少女，黑长直，细瘦身形。\n"
    "黑猫：瘦长、警觉的黑猫，细尾，尖耳。"
)

config = ModuleBLlmConfig(
    provider="siliconflow",
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V4-Flash",
    api_key_file=".secrets/siliconflow_api_key.txt",
    timeout_seconds=120.0,
    retry_times=1,
    temperature=0.7,
    top_p=0.9,
    stream=True,
)

describer = Role1ImageryDescriber(logger=logging.getLogger("bench"), llm_config=config, project_root=PROJECT_ROOT)
t0 = time.perf_counter()
try:
    result = describer.generate(user_template)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"OK | t=0.7 | 意象数: {len(result)} | 耗时: {elapsed:.1f}ms ({elapsed/1000:.2f}s)")
    for item in result:
        print(f"  - {item.imagery_name}: {str(item.pos_zh)[:60]}...")
except Exception as e:
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"FAIL | t=0.7 | 耗时: {elapsed:.1f}ms | 错误: {e}")
