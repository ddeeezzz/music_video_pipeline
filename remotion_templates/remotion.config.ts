/**
 * 文件用途：声明 Remotion 本地模板工程的最小配置。
 * 核心流程：约束输出目录、覆盖文件策略，供本地渲染与后续 Python 调用复用。
 * 输入输出：输入为 Remotion CLI 运行时配置，输出为渲染行为设置。
 * 依赖说明：依赖 @remotion/cli/config 官方 Config API。
 * 维护说明：当前只保留最小配置，避免在正式模板链路早期引入过多工程噪声。
 */

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
 * 功能说明：不指定浏览器路径，让 Remotion 在首次运行时自动下载 Headless Chrome。
 * 参数说明：无。
 * 返回值：无。
 * 异常说明：无。
 * 边界条件：需要可用的网络连接以完成首次浏览器下载。
 */
// 不调用 Config.setBrowserExecutable()，Remotion 会自动下载浏览器
