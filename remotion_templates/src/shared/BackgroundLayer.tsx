/**
 * 文件用途：渲染模板请求中的背景层。
 * 核心流程：根据 kind 选择 none / solid / image / video 的呈现方式。
 * 输入输出：输入为背景请求，输出为铺满画面的背景层 JSX。
 * 依赖说明：依赖 remotion 的 AbsoluteFill、Img、OffthreadVideo。
 * 维护说明：背景由调用方明确传入，本组件不主动补任何审美元素。
 */

// 第三方库：用于构建全屏背景层，并解析 public 静态资源。
import {AbsoluteFill, Img, OffthreadVideo, staticFile} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于提供背景请求类型。
import type {BackgroundRequest} from "../types";

/**
 * 功能说明：把模板请求中的资源路径转换为 Remotion 可消费的静态资源地址。
 * 参数说明：
 * - src: 调用方传入的原始路径。
 * 返回值：
 * - string：可被 Img / OffthreadVideo 使用的地址。
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
 * 功能说明：渲染背景层。
 * 参数说明：
 * - background: 背景请求对象。
 * 返回值：
 * - ReactElement | null：背景层组件；none 时返回纯透明层。
 * 异常说明：无。
 * 边界条件：未知 kind 在类型层已被约束，不在运行时额外兜底。
 */
export const BackgroundLayer = ({
  background
}: {
  background: BackgroundRequest;
}): ReactElement | null => {
  if (background.kind === "none") {
    return <AbsoluteFill />;
  }

  if (background.kind === "solid") {
    return <AbsoluteFill style={{backgroundColor: background.color}} />;
  }

  if (background.kind === "image") {
    return (
      <AbsoluteFill>
        <Img
          src={resolveAssetSrc(background.src)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover"
          }}
        />
      </AbsoluteFill>
    );
  }

  return (
    <AbsoluteFill>
      <OffthreadVideo
        src={resolveAssetSrc(background.src)}
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover"
        }}
      />
    </AbsoluteFill>
  );
};
