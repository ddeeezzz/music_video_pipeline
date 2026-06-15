/**
 * 文件用途：实现全屏横向滚动的 ScrollTemplate 组件。
 * 核心流程：每个 symbol 依次铺满全屏显示，水平滑入/滑出切换。
 * 输入输出：输入为 ScrollTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 维护说明：一次只展示一个 symbol，通过 translateX 实现左右切换过渡。
 */

// 第三方库：用于读取当前帧。
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于渲染背景层。
import {BackgroundLayer} from "../shared/BackgroundLayer";
// 项目内模块：用于渲染歌词叠加层。
import {LyricsOverlay} from "../shared/LyricsOverlay";
// 项目内模块：用于提供模板请求类型。
import type {ScrollTemplateRequest} from "../types";

// 常量：模板画布固定宽高，1344×840。
const TEMPLATE_WIDTH = 1344;
const TEMPLATE_HEIGHT = 840;
// 常量：模板自然动画帧数（保持不变速）。
const NATURAL_FRAMES = 144;
// 常量：循环模式默认单轮跨越拍数。
const DEFAULT_LOOP_BEATS = 4;
// 常量：每个 symbol 时间段中用于过渡切换的比例。
const TRANSITION_RATIO = 0.35;

/**
 * 功能说明：把模板请求中的资源路径转换为 Remotion 可消费的静态资源地址。
 */
const resolveAssetSrc = (src: string): string => {
  const normalized = String(src).trim();
  if (!normalized) return normalized;
  if (normalized.startsWith("/")) return staticFile(normalized);
  return normalized;
};

/**
 * 功能说明：从指定 slot 的 frames 中按局部进度选取当前帧的 src。
 */
const getSlotSrc = (slotIndex: number, localProgress: number, slots: ScrollTemplateRequest["slots"]): string => {
  const slotFrames = slots[slotIndex]?.frames ?? [];
  if (slotFrames.length === 0) return "";
  if (slotFrames.length <= 1) return slotFrames[0].src;
  const safeProgress = Math.min(1, Math.max(0, localProgress));
  const frameIndex = Math.min(Math.floor(safeProgress * slotFrames.length), slotFrames.length - 1);
  return slotFrames[frameIndex].src;
};

/**
 * 功能说明：渲染全屏横向滚动的 ScrollTemplate。
 * 每个 symbol 依次占据全屏，通过水平滑动切换。
 */
export const ScrollTemplate = (props: ScrollTemplateRequest): ReactElement => {
  const frame = useCurrentFrame();
  const animFrame = Math.min(frame, NATURAL_FRAMES - 1);
  const numSlots = Math.max(1, props.slots.length);

  // 计算循环相关参数
  const loopBeats = props.motion.loop_beats ?? DEFAULT_LOOP_BEATS;
  const loopDurationInFrames = Math.max(
    1,
    Math.round((loopBeats * 60 * props.fps) / Math.max(0.001, props.bpm))
  );

  // 根据是否循环决定有效时长和有效帧
  const effectiveDuration = props.motion.loop ? loopDurationInFrames : props.duration_in_frames;
  const effectiveFrame = props.motion.loop ? animFrame % loopDurationInFrames : animFrame;

  // 每个 symbol 的时间段长度
  const framesPerSlot = effectiveDuration / numSlots;
  const transitionFrames = Math.max(1, Math.floor(framesPerSlot * TRANSITION_RATIO));

  // 当前 slot 索引
  const rawSlotIndex = effectiveFrame / framesPerSlot;
  const slotIndex = Math.min(Math.floor(rawSlotIndex), numSlots - 1);
  // 下一个 slot 索引（循环模式下首尾相连）
  const nextSlotIndex = props.motion.loop ? (slotIndex + 1) % numSlots : Math.min(slotIndex + 1, numSlots - 1);

  // 是否处于过渡阶段（最后一个 slot 不过渡，除非循环模式）
  const isLastSlot = !props.motion.loop && slotIndex >= numSlots - 1;

  // 当前 slot 内的局部帧
  const localFrame = effectiveFrame - slotIndex * framesPerSlot;
  const inTransition = !isLastSlot && localFrame >= (framesPerSlot - transitionFrames);

  // 过渡进度和缓动
  let transitionProgress = 0;
  if (inTransition) {
    const rawProgress = (localFrame - (framesPerSlot - transitionFrames)) / transitionFrames;
    // easeInOutCubic
    transitionProgress = rawProgress < 0.5
      ? 4 * rawProgress * rawProgress * rawProgress
      : 1 - Math.pow(-2 * rawProgress + 2, 3) / 2;
  }

  // 当前 symbol 逐渐左移出画，下一个 symbol 从右侧滑入
  const translateXCurrent = inTransition ? -TEMPLATE_WIDTH * transitionProgress : 0;
  const translateXNext = inTransition ? TEMPLATE_WIDTH * (1 - transitionProgress) : TEMPLATE_WIDTH;
  const opacityNext = inTransition ? transitionProgress : 0;

  // 获取当前和下一个 symbol 的图片 src
  const localProgress = localFrame / Math.max(1, framesPerSlot);
  const currentSrc = getSlotSrc(slotIndex, localProgress, props.slots);
  const nextSrc = inTransition ? getSlotSrc(nextSlotIndex, 0, props.slots) : "";

  // 全屏显示比例：固定使用 0.70/0.90，不受预处理裁剪 ratio 影响
  const DISPLAY_WIDTH_RATIO = 0.70;
  const DISPLAY_HEIGHT_RATIO = 0.90;
  const safeWRatio = DISPLAY_WIDTH_RATIO;
  const safeHRatio = DISPLAY_HEIGHT_RATIO;

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
      <LyricsOverlay lyrics={props.lyrics} />
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          overflow: "hidden"
        }}
      >
        {/* 当前 symbol */}
        <Img
          src={resolveAssetSrc(currentSrc)}
          style={{
            width: `${safeWRatio * 100}%`,
            height: `${safeHRatio * 100}%`,
            maxWidth: "100%",
            maxHeight: "100%",
            objectFit: "contain",
            position: "absolute",
            transform: `translateX(${translateXCurrent}px)`
          }}
        />
        {/* 下一个 symbol（过渡期间显示） */}
        {inTransition && (
          <Img
            src={resolveAssetSrc(nextSrc)}
            style={{
              width: `${safeWRatio * 100}%`,
              height: `${safeHRatio * 100}%`,
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              position: "absolute",
              transform: `translateX(${translateXNext}px)`,
              opacity: opacityNext
            }}
          />
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
