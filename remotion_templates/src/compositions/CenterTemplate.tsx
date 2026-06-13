/**
 * 文件用途：实现最小可用的 CenterTemplate 正式模板组件。
 * 核心流程：读取背景请求 -> 按 subject_kind 分类处理：
 *   - object：缓慢旋转一周
 *   - 其他（human/animal/scene）：按 energy_level 做缓慢缩放/平移
 * 输入输出：输入为 CenterTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 依赖说明：依赖 remotion 的当前帧上下文与共享背景层、符号层。
 * 维护说明：energy_level=high 缓慢放大，mid 缓慢左移，low 缓慢缩小。
 */

// 第三方库：用于读取当前帧并构建动画样式。
import {AbsoluteFill, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于渲染背景层。
import {BackgroundLayer} from "../shared/BackgroundLayer";
// 项目内模块：用于渲染中心符号层。
import {SymbolLayer} from "../shared/SymbolLayer";
// 项目内模块：用于提供模板请求类型。
import type {CenterTemplateRequest} from "../types";

// 常量：模板画布固定宽度，1920×1200 16:10 宽屏。
const TEMPLATE_WIDTH = 1920;
// 常量：模板画布固定高度，1920×1200 16:10 宽屏。
const TEMPLATE_HEIGHT = 1200;

/**
 * 功能说明：渲染居中单图模板。
 * 参数说明：
 * - props: CenterTemplate 正式请求。
 * 返回值：
 * - ReactElement：模板画面。
 * 异常说明：无。
 * 边界条件：模板只负责几何布局与时间运动，不主动决定背景审美。
 */
export const CenterTemplate = (props: CenterTemplateRequest): ReactElement => {
  const frame = useCurrentFrame();

  // subject_kind === "object" 时：缓慢旋转效果
  const isObject = (props as any).subject_kind === "object";
  if (isObject) {
    const duration = Math.max(1, props.duration_in_frames);
    const progress = Math.min(1, frame / duration);
    const rotateDeg = 360 * progress;

    const currentSymbol = props.frames[
      Math.min(
        Math.floor((frame / duration) * props.frames.length),
        props.frames.length - 1
      )
    ];

    return (
      <AbsoluteFill
        style={{
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          overflow: "hidden",
          filter: "grayscale(100%)"
        }}
      >
        <BackgroundLayer background={props.background} />
        <AbsoluteFill
          style={{
            transform: `rotate(${rotateDeg}deg)`,
            transformOrigin: "center center"
          }}
        >
          <SymbolLayer
            src={currentSymbol.src}
            widthRatio={currentSymbol.width_ratio}
            heightRatio={currentSymbol.height_ratio}
          />
        </AbsoluteFill>
      </AbsoluteFill>
    );
  }

  // energy_level 控制缩放/平移行为（非 object 走原有逻辑）
  const energyLevel = props.energy_level ?? "high";
  let pulseScale: number;
  let translateX = 0;

  if (energyLevel === "high") {
    // 缓慢持续放大（默认不配置 energy_level 时走此路）
    const progress = Math.min(1, frame / Math.max(1, props.duration_in_frames));
    pulseScale = 1 + progress * (1 / 8);
  } else if (energyLevel === "mid") {
    // 开场已放大 1/12，左边界对齐视野左边界，右边界在视野外
    // 持续左移直到图片右边界对齐视野右边界
    const progress = Math.min(1, frame / Math.max(1, props.duration_in_frames));
    pulseScale = 1 + 1 / 12;
    translateX = -(TEMPLATE_WIDTH * (1 - 1 / pulseScale)) * progress;
  } else {
    // low energy: 开场先放大 1/8，前半段缩回原大小，后半段保持
    const halfFrames = Math.max(1, props.duration_in_frames / 2);
    const progress = Math.min(1, frame / halfFrames);
    pulseScale = 1 + (1 / 8) * (1 - progress);
  }

  // 从 frames 数组按进度取当前帧
  const currentSymbol = props.frames[
    Math.min(
      Math.floor((frame / Math.max(1, props.duration_in_frames)) * props.frames.length),
      props.frames.length - 1
    )
  ];

  return (
    <AbsoluteFill
      style={{
        width: TEMPLATE_WIDTH,
        height: TEMPLATE_HEIGHT,
        overflow: "hidden",
        filter: "grayscale(100%)"
      }}
    >
      <BackgroundLayer background={props.background} />
      <AbsoluteFill
        style={{
          transform: `translateX(${translateX}px) scale(${pulseScale})`,
          transformOrigin: "center center"
        }}
      >
        <SymbolLayer
          src={currentSymbol.src}
          widthRatio={currentSymbol.width_ratio}
          heightRatio={currentSymbol.height_ratio}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
