/**
 * 文件用途：实现 GridTemplate 正式模板组件。
 * 核心流程：读取背景请求 -> 计算三槽位布局 -> 左右格从上往下、中间格从下往上依次入场保持。
 * 输入输出：输入为 GridTemplateRequest，输出为可直接渲染的视频画面 JSX。
 * 维护说明：入场顺序为左→中→右，每格入场后保持静止。
 */

// 第三方库：用于读取当前帧并执行插值动画。
import {AbsoluteFill, interpolate, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于渲染背景层。
import {BackgroundLayer} from "../shared/BackgroundLayer";
// 项目内模块：用于渲染多图条带层。
import {SymbolStripLayer} from "../shared/SymbolStripLayer";
// 项目内模块：用于渲染歌词叠加层。
import {LyricsOverlay} from "../shared/LyricsOverlay";
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

// 常量：模板画布固定宽度，1344×840，与 C 模块单主体帧尺寸对齐。
const TEMPLATE_WIDTH = 1344;
// 常量：模板画布固定高度，1344×840，与 C 模块单主体帧尺寸对齐。
const TEMPLATE_HEIGHT = 840;
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
  const widthList = props.slots.map((slot) => TEMPLATE_WIDTH * (slot.frames[0]?.width_ratio ?? 0.30));
  const heightList = props.slots.map((slot) => TEMPLATE_HEIGHT * (slot.frames[0]?.height_ratio ?? 0.85));
  // 图片容器宽度小于格子宽度时，将每个图片容器在格子内居中偏移
  const centeredLeftList = leftList.map((slotLeft, i) => {
    const imgW = widthList[i] ?? slotWidth;
    return slotLeft + (slotWidth - imgW) / 2;
  });
  // 用实际渲染高度居中，避免 top 与 heightList 不一致导致垂直偏移
  const topList = heightList.map((h) => (TEMPLATE_HEIGHT - h) / 2);
  const totalActiveFrames = Math.max(
    3,
    Math.round(NATURAL_FRAMES * props.motion.active_ratio)
  );
  const enterFrames = Math.max(1, Math.floor(totalActiveFrames / 3));

  // 每个 slot 独立从 frames 数组按本地时间轴进度取当前帧
  const activeSymbolAt = (index: number) => {
    const slotFrames = props.slots[index]?.frames ?? [];
    if (slotFrames.length <= 1) return slotFrames[0] ?? null;
    const localFrame = Math.max(0, animFrame - enterFrames * index);
    const localDuration = props.duration_in_frames - enterFrames * index;
    const progress = localFrame / Math.max(1, localDuration);
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
      <LyricsOverlay lyrics={props.lyrics} />
      {props.slots.map((_slot, index) => {
        const localFrame = Math.max(0, animFrame - enterFrames * index);
        const rawProgress = Math.min(1, localFrame / Math.max(1, enterFrames));
        // easeInOutCubic 缓动
        const eased = rawProgress < 0.5
          ? 4 * rawProgress * rawProgress * rawProgress
          : 1 - Math.pow(-2 * rawProgress + 2, 3) / 2;
        // 方向：左右格从上往下（负→0），中间格从下往上（正→0）
        const direction = index === 1 ? 1 : -1;
        const translateY = interpolate(eased, [0, 1], [direction * props.motion.enter_distance, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp"
        });
        const opacity = interpolate(eased, [0, 0.15, 1], [0, 1, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp"
        });

        const currentSymbol = activeSymbolAt(index);

        return (
          <AbsoluteFill
            key={`${currentSymbol.src}-${index}`}
            style={{
              transform: `translateY(${translateY}px)`,
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
