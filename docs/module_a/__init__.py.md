# `module_a/__init__.py` 说明（证据化）

## 1. 职责与对外影响
- 职责：聚合 `orchestrator/backends/lyrics/segmentation/timing_energy` 的导出符号，形成 `module_a` 包级命名空间。
- 对外影响：
  - 稳定公共入口仅 `run_module_a`。
  - 大量私有符号通过 `test_compat_api` 暴露给测试/迁移链路。

证据：`src/music_video_pipeline/modules/module_a/__init__.py:1-153`

## 2. 入口与调用关系
- 被调用方：`orchestrator._module_a_namespace()` 通过 `importlib.import_module("music_video_pipeline.modules.module_a")` 动态拿到该包命名空间。
  - 证据：`orchestrator.py:29-40`
- 本文件本身无函数入口，核心行为是 `from ... import ...` 与 `__all__` 组装。

## 3. 条件判断与细节
- 公共导出面：`public_api = ["run_module_a"]`。
- 兼容导出面：`test_compat_api = [...]`，覆盖大量私有函数。
- 最终导出：`__all__ = [*public_api, *test_compat_api]`。

证据：`__init__.py:86-153`

## 4. 常量与阈值表
| 项 | 当前值 | 生效位置 | 作用 |
|---|---:|---|---|
| `public_api` | `['run_module_a']` | `__init__.py:87-89` | 限定稳定公共API。 |
| `test_compat_api` | 私有函数白名单列表 | `__init__.py:91-151` | 兼容测试/迁移脚本调用。 |

说明：本文件无“数值型算法阈值常量”。

## 5. 兼容/被弱化思路
- 兼容导出但非稳定公共API：
  - 代码注释明确“以下导出仅用于测试/迁移兼容，后续按计划逐步收缩”。
  - 证据：`__init__.py:91`
- 当前主链未直接使用 `test_compat_api` 列表中的大多数符号：
  - 主执行只依赖 `run_module_a` 及 orchestrator 内部流程；`test_compat_api` 主要由测试引用。
  - 证据：`tests/test_module_a_exports.py:13-52`
