"""
benchmark_role1_speed.py
用途：精确测量模块 B role1 LLM 调用的各阶段耗时。
用法：cd 到项目根目录，运行：
    python -m pytest tests/benchmark_role1_speed.py -v -s --timeout=300
或直接运行：
    python tests/benchmark_role1_speed.py
"""

import json
import logging
import sys
import time
from pathlib import Path

# 确保可以找到项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from music_video_pipeline.config import ModuleBLlmConfig
from music_video_pipeline.modules.module_b.role1_imagery_describer import Role1ImageryDescriber
from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat_detailed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S.%f",
)
logger = logging.getLogger("bench_role1")


def benchmark_call_llm(
    llm_config: ModuleBLlmConfig,
    messages: list[dict[str, str]],
    label: str = "",
) -> dict:
    """精确测量一次 LLM 调用的各阶段耗时。"""
    project_root = PROJECT_ROOT
    api_key_file = project_root / llm_config.api_key_file
    if not api_key_file.exists():
        logger.warning("⚠️  API Key 文件不存在: %s", api_key_file)

    # -- 阶段 1: API Key 读取 --
    t0 = time.perf_counter()
    api_key = (project_root / llm_config.api_key_file).read_text(encoding="utf-8").strip() if api_key_file.exists() else ""
    t1 = time.perf_counter()
    key_read_ms = (t1 - t0) * 1000

    # -- 阶段 2: prompt 拼装耗时(外部已完成，这里只计时大概) --
    t2 = time.perf_counter()

    # -- 阶段 3: 网络请求 + 首 chunk + 全部接收 --
    first_chunk_time = None
    chunk_times = []
    aggregated_len = 0

    def _on_chunk(aggregated_text: str, delta_text: str) -> None:
        nonlocal first_chunk_time, aggregated_len
        now = time.perf_counter()
        if first_chunk_time is None:
            first_chunk_time = now
        chunk_times.append(now)
        aggregated_len = len(aggregated_text)

    t_request_start = time.perf_counter()

    try:
        response = call_module_b_llm_chat_detailed(
            logger=logger,
            llm_config=llm_config,
            messages=messages,
            project_root=project_root,
            on_stream_chunk=_on_chunk,
        )
    except Exception as e:
        t_fail = time.perf_counter()
        return {
            "label": label,
            "success": False,
            "error": str(e),
            "total_ms": (t_fail - t_request_start) * 1000,
            "key_read_ms": key_read_ms,
        }

    t_response_done = time.perf_counter()

    # -- 计算各阶段耗时 --
    total_ms = (t_response_done - t_request_start) * 1000
    first_chunk_ms = (
        (first_chunk_time - t_request_start) * 1000 if first_chunk_time else None
    )
    last_chunk_ms = (
        (chunk_times[-1] - t_request_start) * 1000 if chunk_times else None
    )
    # 首 chunk 之后到全部收完 = 流式传输耗时
    stream_duration_ms = (
        (chunk_times[-1] - first_chunk_time) * 1000
        if first_chunk_time and len(chunk_times) > 1
        else 0
    )
    # 首 chunk 之后到全部收到 = 后续传输时间
    after_first_ms = (
        (t_response_done - first_chunk_time) * 1000 if first_chunk_time else None
    )

    return {
        "label": label,
        "success": True,
        "model": llm_config.model,
        "provider": llm_config.provider,
        "stream": llm_config.stream,
        "total_ms": round(total_ms, 1),
        "first_chunk_ms": round(first_chunk_ms, 1) if first_chunk_ms is not None else None,
        "last_chunk_ms": round(last_chunk_ms, 1) if last_chunk_ms is not None else None,
        "stream_duration_ms": round(stream_duration_ms, 1),
        "after_first_ms": round(after_first_ms, 1) if after_first_ms is not None else None,
        "key_read_ms": round(key_read_ms, 1),
        "chunk_count": len(chunk_times),
        "response_chars": len(response.content),
        "response_content_preview": response.content[:120],
    }


def print_result(result: dict) -> None:
    """格式化输出测速结果。"""
    sep = "-" * 56
    print(f"\n{sep}")
    print(f"  📊 {result.get('label', 'Role1 测速')}")
    print(sep)
    if not result["success"]:
        print(f"  ❌ 失败: {result.get('error', '未知错误')}")
        print(sep)
        return

    total = result["total_ms"]
    model = result["model"]
    provider = result["provider"]

    print(f"  Model:    {model}")
    print(f"  Provider: {provider}")
    print(f"  Stream:   {result['stream']}")
    print(f"")
    print(f"  ⏱  总耗时:         {total:>8.1f} ms  ({total/1000:.2f} s)")
    print(f"  ─────────────────────────────────────")

    fc = result["first_chunk_ms"]
    if fc is not None:
        print(f"  🚀 首 chunk 耗时:    {fc:>8.1f} ms  ({fc/1000:.2f} s)  ← 网络+模型首 Token")
        print(f"  📡 流传输耗时:       {result['stream_duration_ms']:>8.1f} ms")
        print(f"  📦 收到 chunk 数:    {result['chunk_count']:>8}")
        print(f"  📝 返回字符数:       {result['response_chars']:>8}")
        print(f"  💨 首 chunk→完成:    {result['after_first_ms']:>8.1f} ms")
    else:
        print(f"  (非流式模式，无分块数据)")

    print(sep)
    print()


def main() -> None:
    """主入口：运行测速。"""
    project_root = PROJECT_ROOT

    # 尝试加载配置文件，找到 role1 的 LLM 配置
    config_files = [
        project_root / "configs" / "music_windows_4060" / "jieranduhuo.json",
    ]
    
    test_configs = []

    for cfg_file in config_files:
        if cfg_file.exists():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            llm_raw = raw.get("module_b", {}).get("llm", {})
            if llm_raw:
                llm_config = ModuleBLlmConfig(**llm_raw)
                test_configs.append(("当前配置 (jieranduhuo)", llm_config))
                print(f"✅ 已加载配置: {cfg_file.name}")
                print(f"   model: {llm_config.model}")
                print(f"   provider: {llm_config.provider}")
                print(f"   timeout: {llm_config.timeout_seconds}s")
                print(f"   retry: {llm_config.retry_times}")
                print()

                # 再建一个调快参数的对比
                fast_config = ModuleBLlmConfig(
                    provider=llm_raw.get("provider", "siliconflow"),
                    base_url=llm_raw.get("base_url", "https://api.siliconflow.cn/v1"),
                    model=llm_raw.get("model", "Qwen/Qwen2.5-32B-Instruct"),
                    api_key_file=llm_raw.get("api_key_file", ".secrets/siliconflow_api_key.txt"),
                    timeout_seconds=30.0,
                    first_chunk_timeout_seconds=5.0,
                    retry_times=0,
                    temperature=llm_raw.get("temperature", 0.3),
                    top_p=llm_raw.get("top_p", 0.9),
                    stream=True,
                )
                test_configs.append(("调快版 (retry=0, timeout=30s)", fast_config))

    if not test_configs:
        print("❌ 未找到有效的 LLM 配置文件，使用默认配置")
        test_configs.append(("默认配置", ModuleBLlmConfig()))

    # 构造用户模板（简化版，和实际 role1 输入类似）
    user_template_markdown = (
        "## 故事\n"
        "黑猫与少女在空无一人的城市空间里进行带有不安感的捉迷藏。\n\n"
        "## 意象\n"
        "少女：水手服少女，黑长直，细瘦身形。\n"
        "黑猫：瘦长、警觉的黑猫，细尾，尖耳。"
    )

    # 预热：先跑一次普通 LLM 调用（不计时）
    print("🔄 预热中...")
    _, first_config = test_configs[0]
    try:
        describer = Role1ImageryDescriber(
            logger=logger,
            llm_config=first_config,
            project_root=project_root,
        )
        warmup_result = describer.generate(user_template_markdown)
        print(f"   ✅ 预热完成，解析到 {len(warmup_result)} 个意象\n")
    except Exception as e:
        print(f"   ⚠️  预热失败: {e}")
        print(f"   ℹ️  跳过预热，直接跑非预热版本\n")

    # 正式测试
    for label, config in test_configs:
        print(f"🔍 正在测试: {label}")

        describer = Role1ImageryDescriber(
            logger=logger,
            llm_config=config,
            project_root=project_root,
        )

        t_start = time.perf_counter()
        try:
            result = describer.generate(user_template_markdown)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000

            print(f"   ✅ 成功 | 意象数: {len(result)} | 端到端耗时: {elapsed_ms:.1f} ms ({elapsed_ms/1000:.2f} s)")
            for item in result:
                print(f"      - {item.imagery_name}: {item.pos_zh[:40]}...")
            print()

        except Exception as e:
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000
            print(f"   ❌ 失败 | 耗时: {elapsed_ms:.1f} ms | 错误: {e}")
            print()

        # 测一次底层调用，打详细时间
        print(f"🔬 底层 API 调用测速: {label}")
        try:
            prompt_template = (
                project_root / "configs" / "prompts" / "module_b.role1_visual_director.md"
            ).read_text(encoding="utf-8")

            from music_video_pipeline.modules.module_b.prompt_templates import parse_prompt_sections
            system_text, user_text = parse_prompt_sections(prompt_template)
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text.replace("{{User Template}}", user_template_markdown)},
            ]

            detail = benchmark_call_llm(config, messages, label=label)
            print_result(detail)
        except Exception as e:
            print(f"   ❌ 底层测速失败: {e}\n")


if __name__ == "__main__":
    main()
