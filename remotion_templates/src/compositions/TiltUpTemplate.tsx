/**
 * 文件用途：实现 TiltUpTemplate（镜头上移转场，地面→天空）。
 * 核心流程：两个场景上下排列 → 镜头带变加速度上移 → 旧场景下移出画、新场景从上方进入。
 * 输入输出：输入为 TiltUpTemplateRequest，输出为转场视频画面 JSX。
 * 依赖说明：依赖 remotion 的当前帧上下文与共享背景层、符号层。
 */

import {AbsoluteFill, useCurrentFrame} from "remotion";
import type {ReactElement} from "react";
import {BackgroundLayer} from "../shared/BackgroundLayer";
import {SymbolLayer} from "../shared/SymbolLayer";
import {LyricsOverlay} from "../shared/LyricsOverlay";
import type {TiltUpTemplateRequest} from "../types";

const W = 1344;
const H = 840;
const NATURAL_FRAMES = 12;

const easingFns: Record<string, (t: number) => number> = {
  ease_in_out: (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  ease_out: (t) => 1 - Math.pow(1 - t, 3),
  ease_in: (t) => t * t * t,
};

export const TiltUpTemplate = (props: TiltUpTemplateRequest): ReactElement => {
  const frame = useCurrentFrame();

  // 第二张放大 1.125x 居中，图片边界超出视频边界
  const SCALE = 1.125;
  // Phase 1: 快速下移到第二张下边界和视频下边界对齐（移动整高 H 保证首帧不可见）
  const phase1Dist = H;
  // Phase 2: 在放大图内从下到上慢速平移
  const phase2Dist = H * (SCALE - 1);      // 0.125H

  const phase1Frames = Math.min(NATURAL_FRAMES, props.duration_in_frames);
  const phase2Frames = Math.max(0, props.duration_in_frames - phase1Frames);

  const p1Raw = Math.min(1, Math.max(0, Math.min(frame, phase1Frames - 1) / Math.max(1, phase1Frames - 1)));
  const eased = (easingFns[props.motion.easing] ?? easingFns.ease_in_out)(p1Raw);
  const phase1Travel = phase1Dist * eased;

  const p2Raw = phase2Frames > 0
    ? Math.min(1, Math.max(0, (frame - phase1Frames) / phase2Frames))
    : 0;
  const phase2Travel = phase2Frames > 0 ? phase2Dist * p2Raw : 0;

  const ty = phase1Travel + phase2Travel;

  // frames 数组：按进度逐帧切换插值序列
  const afterSymbol = (props.frames && props.frames.length > 0)
    ? props.frames[Math.min(Math.floor((frame / Math.max(1, props.duration_in_frames)) * props.frames.length), props.frames.length - 1)]
    : props.scene_after.symbol;

  return (
    <AbsoluteFill style={{width: W, height: H, overflow: "hidden", filter: "grayscale(100%)"}}>
      <AbsoluteFill style={{transform: `translateY(${ty}px)`}}>
        {/* 新场景在上方（9/8 缩放，首帧不可见，phase1 结束后下边界对齐视口下侧） */}
        <AbsoluteFill style={{top: -(H + H * (SCALE - 1) / 2), width: W, height: H, transform: "scale(1.125)", transformOrigin: "center center"}}>
          <BackgroundLayer background={props.scene_after.background} />
          <SymbolLayer src={afterSymbol.src} widthRatio={afterSymbol.width_ratio} heightRatio={afterSymbol.height_ratio} />
        </AbsoluteFill>
        {/* 旧场景在中间 */}
        <AbsoluteFill style={{top: 0, width: W, height: H}}>
          <BackgroundLayer background={props.scene_before.background} />
          <SymbolLayer src={props.scene_before.symbol.src} widthRatio={props.scene_before.symbol.width_ratio} heightRatio={props.scene_before.symbol.height_ratio} />
        </AbsoluteFill>
      </AbsoluteFill>
      <LyricsOverlay lyrics={props.lyrics} />
    </AbsoluteFill>
  );
};
