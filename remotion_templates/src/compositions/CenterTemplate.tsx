/**
 * 文件用途：实现最小可用的 CenterTemplate 正式模板组件。
 * 核心流程：读取背景请求 -> 按需计算呼吸缩放 -> 将单个符号放在画面中心。
 * 输入输出：输入为 CenterTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 依赖说明：依赖 remotion 的当前帧上下文与共享背景层、符号层。
 * 维护说明：当前只验证“正式 request JSON -> mp4”主闭环，不在此阶段追求复杂动画。
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

/**
 * 功能说明：根据 BPM 生成轻微节拍缩放比例。
 * 参数说明：
 * - frame: 当前帧序号。
 * - fps: 当前渲染帧率。
 * - bpm: 当前音乐 BPM。
 * - breathe: 是否启用呼吸动画。
 * 返回值：
 * - number：当前帧的缩放倍率。
 * 异常说明：无。
 * 边界条件：breathe=false 时返回 1，表示保持静止显示。
 */
const getBeatPulseScale = (
  frame: number,
  fps: number,
  bpm: number,
  breathe: boolean
): number => {
  if (!breathe) {
    return 1;
  }

  const beatsPerSecond = Math.max(0.001, bpm / 60);
  const phase = (frame / Math.max(1, fps)) * beatsPerSecond * Math.PI * 2;
  const normalizedWave = (Math.sin(phase - Math.PI / 2) + 1) / 2;
  const easedWave = Math.pow(normalizedWave, 2.2);
  return 1 + 0.04 * easedWave;
};

// 常量：模板画布固定宽度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_WIDTH = 512;
// 常量：模板画布固定高度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_HEIGHT = 320;
// 常量：模板自然动画帧数（保持不变速）。
const NATURAL_FRAMES = 48;
// 常量：mid energy 左移放大倍率（仿 PanRightTemplate 第二阶段的放大处理）。
const MID_SCALE = 1.125;

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
  // 保持自然速度：动画帧 clamped 到 NATURAL_FRAMES，之后维持末帧
  const animFrame = Math.min(frame, NATURAL_FRAMES - 1);

  // energy_level 控制缩放/平移行为
  const energyLevel = props.energy_level ?? "high";
  let pulseScale: number;
  let translateX = 0;

  if (energyLevel === "high") {
    pulseScale = getBeatPulseScale(animFrame, props.fps, props.bpm, props.motion.breathe);
  } else if (energyLevel === "mid") {
    // 先快速放大到 MID_SCALE，再在放大图内慢速左移
    const phase1Frames = Math.min(12, props.duration_in_frames);
    const phase2Frames = Math.max(0, props.duration_in_frames - phase1Frames);

    const p1Raw = phase1Frames > 1
      ? Math.min(1, Math.max(0, Math.min(frame, phase1Frames - 1) / (phase1Frames - 1)))
      : 1;
    const eased = p1Raw < 0.5 ? 4 * p1Raw * p1Raw * p1Raw : 1 - Math.pow(-2 * p1Raw + 2, 3) / 2;
    pulseScale = 1 + (MID_SCALE - 1) * eased;

    if (phase2Frames > 0) {
      const p2Raw = Math.min(1, Math.max(0, (frame - phase1Frames) / phase2Frames));
      // 放大后左右各多出 W*(SCALE-1)/2 像素，左移此距离让右侧内容逐渐可见
      translateX = -(TEMPLATE_WIDTH * (MID_SCALE - 1) / 2) * p2Raw;
    }
  } else {
    const progress = Math.min(1, frame / Math.max(1, props.duration_in_frames));
    // low energy: 缓缓缩小 1/8
    pulseScale = 1 - progress * (1 / 8);
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
