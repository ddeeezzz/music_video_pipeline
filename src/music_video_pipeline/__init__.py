"""
文件用途：MVP 音画同步流水线的核心包初始化。
核心流程：对外暴露版本信息，供 CLI 与测试引用。
输入输出：无输入，输出版本字符串常量。
依赖说明：仅依赖 Python 标准语义。
维护说明：版本号与 pyproject.toml 保持一致。
"""

__version__ = "0.1.0"

import os
from pathlib import Path

# 强制设置本地化与离线模式环境变量，并将系统级缓存统一重定向到项目 models 目录
# 此逻辑放在 __init__.py 可确保无论通过 CLI 还是测试/脚本启动，缓存都能自闭环
_workspace_root = Path(__file__).resolve().parents[2]
_models_dir = _workspace_root / "models"

os.environ["HF_HOME"] = str(_models_dir / "audio" / "hf_cache")
os.environ["MODELSCOPE_CACHE"] = str(_models_dir / "audio" / "modelscope")
os.environ["TORCH_HOME"] = str(_models_dir / "audio" / "torch")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"
