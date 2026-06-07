/**
 * 文件用途：实现最小可用的 GridTemplate 正式模板组件。
 * 核心流程：读取背景请求 -> 计算三槽位布局 -> 按总激活时长比例依次执行跳出动画。
 * 输入输出：输入为 GridTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 依赖说明：依赖 remotion 当前帧上下文、插值工具，以及共享背景层与多图条带层。
 * 维护说明：当前只验证“三图横排 + 依次进入”的最小正式闭环，不在此阶段引入更多模板语言。
 */

// 第三方库：用于读取当前帧并执行插值动画。
import {AbsoluteFill, interpolate, spring, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于渲染背景层。
import {BackgroundLayer} from "../shared/BackgroundLayer";
// 项目内模块：用于渲染多图条带层。
import {SymbolStripLayer} from "../shared/SymbolStripLayer";
// 项目内模块：用于提供模板请求类型。
import type {GridTemplateRequest} from "../types";

/**
 * 功能说明：根据方向计算三个槽位的左侧像素位置。
 * 参数说明：
 * - width: 画面宽度。
 * - slotWidth: 单槽位宽度。
 * 返回值：
 * - number[]：按符号顺序排列的左侧像素位置数组。
 * 异常说明：无。
 * 边界条件：三槽位整体始终保持居中。
 */
const getSlotLeftList = (
  width: number,
  slotWidth: number
): number[] => {
  const totalWidth = slotWidth * 3;
  const startLeft = (width - totalWidth) / 2;
  return [0, 1, 2].map((index) => startLeft + index * slotWidth);
};

// 常量：模板画布固定宽度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_WIDTH = 512;
// 常量：模板画布固定高度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_HEIGHT = 320;
// 常量：模板自然动画帧数（保持不变速）。
const NATURAL_FRAMES = 84;

/**
 * 功能说明：渲染三图依次进入的网格模板。
 * 参数说明：
 * - props: GridTemplate 正式请求。
 * 返回值：
 * - ReactElement：模板画面。
 * 异常说明：无。
 * 边界条件：当前只支持 3 个符号的固定条带布局。
 */
export const GridTemplate = (props: GridTemplateRequest): ReactElement => {
  const frame = useCurrentFrame();
  // 保持自然速度：动画帧 clamped 到 NATURAL_FRAMES，之后维持末帧
  const animFrame = Math.min(frame, NATURAL_FRAMES - 1);

  const visibleCellCount = Math.max(1, props.layout.visible_cell_count);
  const slotWidth = TEMPLATE_WIDTH / visibleCellCount;
  const leftList = getSlotLeftList(TEMPLATE_WIDTH, slotWidth);
  const topList = props.slots.map(
    (slot) => (TEMPLATE_HEIGHT - TEMPLATE_HEIGHT * (slot.frames[0]?.height_ratio ?? 0.52)) / 2
  );
  const widthList = props.slots.map((slot) => TEMPLATE_WIDTH * (slot.frames[0]?.width_ratio ?? 0.26));
  const heightList = props.slots.map((slot) => TEMPLATE_HEIGHT * (slot.frames[0]?.height_ratio ?? 0.52));
  // 图片容器宽度小于格子宽度时，将每个图片容器在格子内居中偏移
  const centeredLeftList = leftList.map((slotLeft, i) => {
    const imgW = widthList[i] ?? slotWidth;
    return slotLeft + (slotWidth - imgW) / 2;
  });
  const totalActiveFrames = Math.max(
    3,
    Math.round(NATURAL_FRAMES * props.motion.active_ratio)
  );
  const enterFrames = Math.max(1, Math.floor(totalActiveFrames / 3));

  // 每个 slot 独立从 frames 数组按进度取当前帧
  const activeSymbolAt = (index: number) => {
    const slotFrames = props.slots[index]?.frames ?? [];
    if (slotFrames.length <= 1) return slotFrames[0] ?? null;
    const progress = animFrame / Math.max(1, props.duration_in_frames);
    return slotFrames[Math.min(Math.floor(progress * slotFrames.length), slotFrames.length - 1)];
  };

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
      {props.slots.map((_slot, index) => {
        const localFrame = Math.max(0, animFrame - enterFrames * index);
        const progress = spring({
          fps: props.fps,
          frame: localFrame,
          durationInFrames: enterFrames
        });
        const overshootScale = 1 + props.motion.overshoot_ratio;
        const scale = interpolate(progress, [0, 0.8, 1], [1.2, 1.0, 1.0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp"
        });
        const translateY = interpolate(progress, [0, 1], [props.motion.enter_distance, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp"
        });
        const opacity = interpolate(progress, [0, 0.15, 1], [0, 0.65, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp"
        });

        const currentSymbol = activeSymbolAt(index);

        return (
          <AbsoluteFill
            key={`${currentSymbol.src}-${index}`}
            style={{
              transform: `translateY(${translateY}px) scale(${scale})`,
              transformOrigin: "center center",
              opacity
            }}
          >
            <SymbolStripLayer
              symbolSrcList={[currentSymbol.src]}
              leftList={[centeredLeftList[index]]}
              top={0}
              width={slotWidth}
              height={TEMPLATE_HEIGHT}
              topList={[topList[index]]}
              widthList={[widthList[index]]}
              heightList={[heightList[index]]}
            />
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};
