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
  const visibleCellCount = Math.max(1, props.layout.visible_cell_count);
  const slotWidth = TEMPLATE_WIDTH / visibleCellCount;
  const leftList = getSlotLeftList(TEMPLATE_WIDTH, slotWidth);
  const topList = props.symbols.map(
    (symbol) => (TEMPLATE_HEIGHT - TEMPLATE_HEIGHT * symbol.height_ratio) / 2
  );
  const widthList = props.symbols.map((symbol) => TEMPLATE_WIDTH * symbol.width_ratio);
  const heightList = props.symbols.map((symbol) => TEMPLATE_HEIGHT * symbol.height_ratio);
  const totalActiveFrames = Math.max(
    3,
    Math.round(props.duration_in_frames * props.motion.active_ratio)
  );
  const enterFrames = Math.max(1, Math.floor(totalActiveFrames / 3));

  return (
    <AbsoluteFill
      style={{
        width: TEMPLATE_WIDTH,
        height: TEMPLATE_HEIGHT,
        overflow: "hidden"
      }}
    >
      <BackgroundLayer background={props.background} />
      {props.symbols.map((symbol, index) => {
        const localFrame = Math.max(0, frame - enterFrames * index);
        const progress = spring({
          fps: props.fps,
          frame: localFrame,
          durationInFrames: enterFrames
        });
        const overshootScale = 1 + props.motion.overshoot_ratio;
        const scale = interpolate(progress, [0, 0.8, 1], [0.75, overshootScale, 1], {
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

        return (
          <AbsoluteFill
            key={`${symbol.src}-${index}`}
            style={{
              transform: `translateY(${translateY}px) scale(${scale})`,
              transformOrigin: "center center",
              opacity
            }}
          >
            <SymbolStripLayer
              symbolSrcList={[symbol.src]}
              leftList={[leftList[index]]}
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
