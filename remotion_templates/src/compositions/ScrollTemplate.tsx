/**
 * 文件用途：实现最小可用的 ScrollTemplate 正式模板组件。
 * 核心流程：读取背景请求 -> 按格子数计算静态槽位 -> 根据循环开关决定匀速循环或全段滚动。
 * 输入输出：输入为 ScrollTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 依赖说明：依赖 remotion 当前帧上下文，以及共享背景层与多图条带层。
 * 维护说明：当前只验证“连续循环条带”主闭环，不在此阶段叠加更多镜头语言。
 */

// 第三方库：用于读取当前帧。
import {AbsoluteFill, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于渲染背景层。
import {BackgroundLayer} from "../shared/BackgroundLayer";
// 项目内模块：用于渲染多图条带层。
import {SymbolStripLayer} from "../shared/SymbolStripLayer";
// 项目内模块：用于提供模板请求类型。
import type {ScrollTemplateRequest} from "../types";

/**
 * 功能说明：计算当前帧对应的条带横向滚动位移。
 * 参数说明：
 * - frame: 当前帧序号。
 * - durationInFrames: 总帧数。
 * - travelDistance: 需要横向滚动的总位移。
 * - loop: 是否启用循环滚动。
 * - loopDurationInFrames: 单轮循环时长。
 * 返回值：
 * - number：相对初始条带位置的横向位移像素值。
 * 异常说明：无。
 * 边界条件：方向当前固定为从右向左。
 */
const getTranslateX = (
  frame: number,
  durationInFrames: number,
  travelDistance: number,
  loop: boolean,
  loopDurationInFrames: number
): number => {
  if (loop) {
    const safeLoopDuration = Math.max(1, loopDurationInFrames);
    const cycleProgress = (frame % safeLoopDuration) / safeLoopDuration;
    return -travelDistance * cycleProgress;
  }

  const maxFrame = Math.max(1, durationInFrames - 1);
  const progress = Math.min(1, Math.max(0, frame / maxFrame));
  return -travelDistance * progress;
};

// 常量：模板画布固定宽度，1920×1200 16:10 宽屏。
const TEMPLATE_WIDTH = 1920;
// 常量：模板画布固定高度，1920×1200 16:10 宽屏。
const TEMPLATE_HEIGHT = 1200;
// 常量：模板自然动画帧数（保持不变速）。
const NATURAL_FRAMES = 144;
// 常量：循环模式默认单轮跨越拍数，后续可继续改为音频特征直接驱动。
const DEFAULT_LOOP_BEATS = 4;

/**
 * 功能说明：渲染连续循环条带模板。
 * 参数说明：
 * - props: ScrollTemplate 正式请求。
 * 返回值：
 * - ReactElement：模板画面。
 * 异常说明：无。
 * 边界条件：模板只负责几何布局与时间运动，不主动决定背景审美。
 */
export const ScrollTemplate = (props: ScrollTemplateRequest): ReactElement => {
  const frame = useCurrentFrame();
  // 保持自然速度：动画帧 clamped 到 NATURAL_FRAMES，之后维持末帧
  const animFrame = Math.min(frame, NATURAL_FRAMES - 1);

  // 每个 slot 独立从 frames 数组按进度取当前帧
  const currentSrcAt = (index: number): string => {
    const slotFrames = props.slots[index]?.frames ?? [];
    if (slotFrames.length <= 1) return slotFrames[0]?.src ?? "";
    const progress = animFrame / Math.max(1, props.duration_in_frames);
    return slotFrames[Math.min(Math.floor(progress * slotFrames.length), slotFrames.length - 1)].src;
  };

  const visibleCellCount = Math.max(1, props.layout.visible_cell_count);
  const cellWidth = TEMPLATE_WIDTH / visibleCellCount;
  const slotWidth = cellWidth;
  const step = slotWidth;
  const firstStripWidth = step * props.slots.length;
  const loopedSrcList = [
    ...props.slots.map((_, i) => currentSrcAt(i)),
    ...props.slots.map((_, i) => currentSrcAt(i)),
  ];
  const totalWidth = step * loopedSrcList.length;
  const baseLeft = (TEMPLATE_WIDTH - firstStripWidth) / 2;
  const topList = props.slots.map(
    (slot) => (TEMPLATE_HEIGHT - TEMPLATE_HEIGHT * (slot.frames[0]?.height_ratio ?? 0.52)) / 2
  );
  const widthList = props.slots.map((slot) => TEMPLATE_WIDTH * (slot.frames[0]?.width_ratio ?? 0.26));
  const heightList = props.slots.map((slot) => TEMPLATE_HEIGHT * (slot.frames[0]?.height_ratio ?? 0.52));
  const leftList = loopedSrcList.map((_, index) => baseLeft + index * step);
  const loopBeats = props.motion.loop_beats ?? DEFAULT_LOOP_BEATS;
  const loopDurationInFrames = Math.max(
    1,
    Math.round((loopBeats * 60 * props.fps) / Math.max(0.001, props.bpm))
  );
  const translateX = getTranslateX(
    animFrame,
    NATURAL_FRAMES,
    firstStripWidth,
    props.motion.loop,
    loopDurationInFrames
  );

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
          width: totalWidth,
          left: 0,
          transform: `translateX(${translateX}px)`
        }}
      >
        <SymbolStripLayer
          symbolSrcList={loopedSrcList}
          leftList={leftList}
          top={0}
          width={slotWidth}
          height={TEMPLATE_HEIGHT}
          topList={[...topList, ...topList]}
          widthList={[...widthList, ...widthList]}
          heightList={[...heightList, ...heightList]}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
