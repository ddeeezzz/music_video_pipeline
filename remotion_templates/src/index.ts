/**
 * 文件用途：作为 Remotion 工程入口，注册 Root 组件。
 * 核心流程：加载 Root -> 调用 registerRoot -> 暴露全部 Composition。
 * 输入输出：输入为 Remotion CLI 入口调用，输出为已注册的根组件树。
 * 依赖说明：依赖 remotion 的 registerRoot 与本地 Root 组件。
 * 维护说明：本文件应保持极薄，只承担入口职责。
 */

// 第三方库：用于向 Remotion 注册根组件。
import {registerRoot} from "remotion";
// 项目内模块：用于承载全部模板 Composition 注册。
import {Root} from "./Root";

/**
 * 功能说明：向 Remotion 注册根组件。
 * 参数说明：无。
 * 返回值：无。
 * 异常说明：无。
 * 边界条件：Root 必须是纯组件，不在入口层掺入业务状态。
 */
registerRoot(Root);
