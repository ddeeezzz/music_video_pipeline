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
// 项目内模块：用于承载 TiltUpTemplate 组件实现。
import {TiltUpTemplate} from "./compositions/TiltUpTemplate";
// 项目内模块：用于承载 TiltDownTemplate 组件实现。
import {TiltDownTemplate} from "./compositions/TiltDownTemplate";
// 项目内模块：用于承载 PanRightTemplate 组件实现。
import {PanRightTemplate} from "./compositions/PanRightTemplate";
// 项目内模块：用于提供 Studio Props 面板所需 schema。
import {
  centerTemplateSchema,
  gridTemplateSchema,
  scrollTemplateSchema,
  tiltUpTemplateSchema,
  tiltDownTemplateSchema,
  panRightTemplateSchema
} from "./schema";

// 常量：模板画布固定宽度，1344×840，与 C 模块单主体帧尺寸对齐。
const TEMPLATE_WIDTH = 1344;
// 常量：模板画布固定高度，1344×840，与 C 模块单主体帧尺寸对齐。
const TEMPLATE_HEIGHT = 840;
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
        defaultProps={{"template":"center" as const,"fps":24,"duration_in_frames":48,"bpm":130,"background":{"kind":"solid" as const,"color":"#FFFFFF"},"frames":[{"src":"/fixtures/center-symbol.svg","width_ratio":0.42,"height_ratio":0.42}],"motion":{"breathe":true}}}
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
          slots: [
            {frames: [{src: "/fixtures/grid-a.svg", width_ratio: 0.30, height_ratio: 0.85}]},
            {frames: [{src: "/fixtures/grid-b.svg", width_ratio: 0.30, height_ratio: 0.85}]},
            {frames: [{src: "/fixtures/grid-c.svg", width_ratio: 0.30, height_ratio: 0.85}]}
          ],
          layout: {visible_cell_count: 3},
          motion: {active_ratio: 0.45, overshoot_ratio: 0.08, enter_distance: 240}
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
          slots: [
            {frames: [{src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72}]},
            {frames: [{src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72}]},
            {frames: [{src: "/fixtures/scroll-symbol.svg", width_ratio: 0.28, height_ratio: 0.72}]}
          ],
          layout: {visible_cell_count: 3},
          motion: {
            loop: false
          }
        }}
      />
      <Composition
        id="TiltUpTemplate"
        component={TiltUpTemplate}
        schema={tiltUpTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{
          template: "tilt_up" as const,
          fps: 24,
          duration_in_frames: 12,
          bpm: 130,
          scene_before: {
            background: {kind: "solid" as const, color: "#FFFFFF"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          scene_after: {
            background: {kind: "solid" as const, color: "#FFFF00"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          motion: {travel_px: 1080, easing: "ease_in_out" as const}
        }}
      />
      <Composition
        id="TiltDownTemplate"
        component={TiltDownTemplate}
        schema={tiltDownTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{
          template: "tilt_down" as const,
          fps: 24,
          duration_in_frames: 12,
          bpm: 130,
          scene_before: {
            background: {kind: "solid" as const, color: "#FFFFFF"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          scene_after: {
            background: {kind: "solid" as const, color: "#FFFF00"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          motion: {travel_px: 1080, easing: "ease_in_out" as const}
        }}
      />
      <Composition
        id="PanRightTemplate"
        component={PanRightTemplate}
        schema={panRightTemplateSchema}
        calculateMetadata={({props}) => ({
          width: TEMPLATE_WIDTH,
          height: TEMPLATE_HEIGHT,
          fps: props.fps,
          durationInFrames: props.duration_in_frames
        })}
        defaultProps={{
          template: "pan_right" as const,
          fps: 24,
          duration_in_frames: 12,
          bpm: 130,
          scene_before: {
            background: {kind: "solid" as const, color: "#FFFFFF"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          scene_after: {
            background: {kind: "solid" as const, color: "#FFFF00"},
            symbol: {src: "/fixtures/center-symbol.svg", width_ratio: 0.42, height_ratio: 0.42}
          },
          motion: {travel_px: 1920, easing: "ease_in_out" as const}
        }}
      />
    </>
  );
};
