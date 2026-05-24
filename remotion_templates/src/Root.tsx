/**
 * 文件用途：注册模板工程的全部 Composition。
 * 核心流程：定义默认 props -> 注册各正式模板 Composition，并允许 Studio 回写默认值。
 * 输入输出：输入为 Remotion Studio / CLI 的组合加载过程，输出为 Composition 注册结果。
 * 依赖说明：依赖 remotion Composition、模板组件与正式类型定义。
 * 维护说明：当前阶段注册 CenterTemplate、GridTemplate 与 ScrollTemplate。
 */

// 第三方库：用于注册 Remotion Composition。
import {Composition} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于承载 CenterTemplate 组件实现。
import {CenterTemplate} from "./compositions/CenterTemplate";
// 项目内模块：用于承载 GridTemplate 组件实现。
import {GridTemplate} from "./compositions/GridTemplate";
// 项目内模块：用于承载 ScrollTemplate 组件实现。
import {ScrollTemplate} from "./compositions/ScrollTemplate";
// 项目内模块：用于提供 Studio Props 面板所需 schema。
import {
  centerTemplateSchema,
  gridTemplateSchema,
  scrollTemplateSchema
} from "./schema";

// 常量：模板画布固定宽度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_WIDTH = 512;
// 常量：模板画布固定高度，和 Tooncrafter 推荐分辨率对齐。
const TEMPLATE_HEIGHT = 320;
/**
 * 功能说明：注册当前模板工程的全部 Composition。
 * 参数说明：无。
 * 返回值：
 * - ReactElement：Composition 注册树。
 * 异常说明：无。
 * 边界条件：后续新增 grid / scroll 时继续在此处串联注册。
 */
export const Root = (): ReactElement => {
  return (
    <>
      <Composition
        id="CenterTemplate"
        component={CenterTemplate}
        schema={centerTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{"template":"center" as const,"fps":24,"duration_in_frames":48,"bpm":130,"background":{"kind":"solid" as const,"color":"#FFFFFF"},"symbol":{"src":"/fixtures/center-symbol.svg","width_ratio":0.42,"height_ratio":0.42},"motion":{"breathe":true}}}
      />
      <Composition
        id="GridTemplate"
        component={GridTemplate}
        schema={gridTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{
          template: "grid" as const,
          fps: 24,
          duration_in_frames: 84,
          bpm: 130,
          background: {kind: "solid" as const, color: "#FFFFFF"},
          symbols: [
            {src: "/fixtures/grid-a.svg", width_ratio: 0.26, height_ratio: 0.52},
            {src: "/fixtures/grid-b.svg", width_ratio: 0.26, height_ratio: 0.52},
            {src: "/fixtures/grid-c.svg", width_ratio: 0.26, height_ratio: 0.52}
          ],
          layout: {visible_cell_count: 3},
          motion: {active_ratio: 0.45, overshoot_ratio: 0.08, enter_distance: 72}
        }}
      />
      <Composition
        id="ScrollTemplate"
        component={ScrollTemplate}
        schema={scrollTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{
          template: "scroll",
          fps: 24,
          duration_in_frames: 144,
          bpm: 130,
          background: {
            kind: "solid",
            color: "#FFFFFF"
          },
          symbols: [
            {src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72},
            {src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72},
            {src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72}
          ],
          layout: {visible_cell_count: 3},
          motion: {
            loop: false
          }
        }}
      />
    </>
  );
};
