/**
 * 文件用途：渲染横向排列的多个视觉符号条带。
 * 核心流程：根据槽位宽度、间距与位置列表，将多个图片按统一规则放入对应区域。
 * 输入输出：输入为符号路径数组与槽位布局参数，输出为一组绝对定位的图片层 JSX。
 * 依赖说明：依赖 remotion AbsoluteFill 与 Img。
 * 维护说明：本组件只负责多图排布，不负责时间动画与背景逻辑。
 */

// 第三方库：用于构建绝对定位图片层，并解析 public 静态资源。
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
 * 功能说明：渲染横向条带上的多个符号。
 * 参数说明：
 * - symbolSrcList: 符号路径数组。
 * - leftList: 每个槽位的左侧像素位置。
 * - top: 槽位顶部像素位置。
 * - width: 单个槽位宽度。
 * - height: 单个槽位高度。
 * - topList: 可选的逐项顶部位置覆盖列表。
 * - widthList: 可选的逐项宽度覆盖列表。
 * - heightList: 可选的逐项高度覆盖列表。
 * 返回值：
 * - ReactElement：符号条带层。
 * 异常说明：无。
 * 边界条件：图片保持原始宽高比并完整包含于槽位内。
 */
export const SymbolStripLayer = ({
  symbolSrcList,
  leftList,
  top,
  width,
  height,
  topList,
  widthList,
  heightList
}: {
  symbolSrcList: string[];
  leftList: number[];
  top: number;
  width: number;
  height: number;
  topList?: number[];
  widthList?: number[];
  heightList?: number[];
}): ReactElement => {
  return (
    <AbsoluteFill>
      {symbolSrcList.map((src, index) => {
        const left = leftList[index] ?? 0;
        const resolvedTop = topList?.[index] ?? top;
        const resolvedWidth = widthList?.[index] ?? width;
        const resolvedHeight = heightList?.[index] ?? height;
        return (
          <AbsoluteFill
            key={`${src}-${index}`}
            style={{
              left,
              top: resolvedTop,
              width: resolvedWidth,
              height: resolvedHeight,
              justifyContent: "center",
              alignItems: "center"
            }}
          >
            <Img
              src={resolveAssetSrc(src)}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain"
              }}
            />
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};
