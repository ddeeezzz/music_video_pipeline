/**
 * 文件用途：声明 Remotion 本地模板工程的最小配置。
 * 核心流程：约束输出目录、覆盖文件策略与本地浏览器路径，供本地渲染与后续 Python 调用复用。
 * 输入输出：输入为 Remotion CLI 运行时配置，输出为渲染行为设置。
 * 依赖说明：依赖 @remotion/cli/config 官方 Config API 与 Node 路径工具。
 * 维护说明：当前只保留最小配置，避免在正式模板链路早期引入过多工程噪声。
 */

// 标准库：用于拼接本地浏览器可执行文件绝对路径。
import path from "node:path";
// 第三方库：用于声明 Remotion CLI 的工程配置。
import {Config} from "@remotion/cli/config";

/**
 * 功能说明：设置模板工程默认输出目录。
 * 参数说明：无。
 * 返回值：无。
 * 异常说明：无。
 * 边界条件：目录不存在时由 Remotion 在渲染时自动创建。
 */
Config.setOutputLocation("out");

/**
 * 功能说明：允许重复渲染时覆盖既有输出文件。
 * 参数说明：无。
 * 返回值：无。
 * 异常说明：无。
 * 边界条件：仅影响本地开发阶段的默认行为。
 */
Config.setOverwriteOutput(true);

/**
 * 功能说明：固定使用项目内已下载的 Chrome Headless Shell，避免运行时自动联网下载。
 * 参数说明：无。
 * 返回值：无。
 * 异常说明：无。
 * 边界条件：浏览器文件缺失时，Remotion 会在实际运行时报出路径错误。
 */
Config.setBrowserExecutable(
  path.resolve(
    process.cwd(),
    "browser",
    "chrome-headless-shell-win64",
    "chrome-headless-shell.exe"
  )
);
