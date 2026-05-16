/**
 * 文件用途：渲染位于模板主层的单个视觉符号。
 * 核心流程：根据尺寸比例把图片放到画面中心，并保持透明背景与原始宽高比。
 * 输入输出：输入为符号路径与尺寸比例，输出为居中的视觉层 JSX。
 * 依赖说明：依赖 remotion AbsoluteFill 与 Img。
 * 维护说明：本组件只负责单图摆放，不承担额外装饰职责。
 */

// 第三方库：用于构建居中符号层，并解析 public 静态资源。
import {AbsoluteFill, Img, staticFile} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";

/**
 * 功能说明：把模板请求中的资源路径转换为 Remotion 可消费的静态资源地址。
 * 参数说明：
 * - src: 调用方传入的原始路径。
 * 返回值：
 * - string：可被 Img 使用的地址。
 * 异常说明：无。
 * 边界条件：以 `/` 开头的路径视为 public 目录资源；其他路径原样透传。
 */
const resolveAssetSrc = (src: string): string => {
  const normalized = String(src).trim();
  if (!normalized) {
    return normalized;
  }

  if (normalized.startsWith("/")) {
    return staticFile(normalized);
  }

  return normalized;
};

/**
 * 功能说明：渲染单个居中视觉符号。
 * 参数说明：
 * - src: 视觉符号路径。
 * - widthRatio: 基于画面宽度的目标宽度比例。
 * - heightRatio: 基于画面高度的目标高度比例。
 * 返回值：
 * - ReactElement：居中的符号层。
 * 异常说明：无。
 * 边界条件：比例应由调用方控制在合理区间内，本组件不额外裁切。
 */
export const SymbolLayer = ({
  src,
  widthRatio,
  heightRatio
}: {
  src: string;
  widthRatio: number;
  heightRatio: number;
}): ReactElement => {
  const safeWidthRatio = Number.isFinite(widthRatio) ? Math.max(0.05, widthRatio) : 0.42;
  const safeHeightRatio = Number.isFinite(heightRatio) ? Math.max(0.05, heightRatio) : 0.42;

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center"
      }}
    >
      <Img
        src={resolveAssetSrc(src)}
        style={{
          width: `${safeWidthRatio * 100}%`,
          height: `${safeHeightRatio * 100}%`,
          maxWidth: "100%",
          maxHeight: "100%",
          objectFit: "contain"
        }}
      />
    </AbsoluteFill>
  );
};
