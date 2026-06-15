/**
 * 文件用途：定义正式模板请求 JSON 的最小 TypeScript 契约。
 * 核心流程：声明背景、符号、布局、运动等结构，供 Composition 与 fixtures 共用。
 * 输入输出：输入为模板请求 JSON，输出为静态类型约束。
 * 依赖说明：仅依赖 TypeScript 类型系统。
 * 维护说明：当前覆盖 CenterTemplate、GridTemplate 与 ScrollTemplate 的最小正式闭环。
 */

/**
 * 功能说明：定义空背景请求。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：none 表示模板层不主动渲染背景。
 */
export type BackgroundNone = {
  kind: "none";
};

/**
 * 功能说明：定义纯色背景请求。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：color 必须由调用方明确传入。
 */
export type BackgroundSolid = {
  kind: "solid";
  color: string;
};

/**
 * 功能说明：定义图片背景请求。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：src 可指向静态资源路径或外部 URL。
 */
export type BackgroundImage = {
  kind: "image";
  src: string;
};

/**
 * 功能说明：定义视频背景请求。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：src 可指向静态资源路径或外部 URL。
 */
export type BackgroundVideo = {
  kind: "video";
  src: string;
};

/**
 * 功能说明：统一背景请求类型。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：当前最小版本只支持 4 种背景形态。
 */
export type BackgroundRequest =
  | BackgroundNone
  | BackgroundSolid
  | BackgroundImage
  | BackgroundVideo;

/**
 * 功能说明：定义单个视觉符号请求。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：width_ratio 与 height_ratio 以画面比例表达，用于暴露最终贴图大小调节入口。
 */
export type SymbolRequest = {
  src: string;
  width_ratio: number;
  height_ratio: number;
};

/**
 * 功能说明：定义单行歌词数据。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：start_frame/end_frame 为相对模板时间轴的帧序号。
 */
export type LyricItem = {
  text: string;
  translated_text?: string;
  start_frame: number;
  end_frame: number;
};

/**
 * 功能说明：定义 CenterTemplate 运动参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：breathe 为 true 时启用呼吸动画；频次当前仍由 BPM 驱动。
 */
export type CenterMotionRequest = {
  breathe: boolean;
};

/**
 * 功能说明：定义 CenterTemplate 的正式请求结构。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：duration_in_frames 必须由调用方按时间轴精确给定。
 */
export type CenterTemplateRequest = {
  template: "center";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  background: BackgroundRequest;
  frames: SymbolRequest[];
  motion: CenterMotionRequest;
  energy_level?: "low" | "mid" | "high";
  rhythm_tension?: number;
  lyrics?: LyricItem[];
  font_path?: string;
};

/**
 * 功能说明：定义 GridTemplate 的布局参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：visible_cell_count 表示一张完整静态图应占多少等分格子。
 */
export type GridLayoutRequest = {
  visible_cell_count: number;
};

/**
 * 功能说明：定义 GridTemplate 的运动参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：active_ratio 表示三格完成跳出所占的总时长比例。
 */
export type GridMotionRequest = {
  active_ratio: number;
  overshoot_ratio: number;
  enter_distance: number;
};

/**
 * 功能说明：定义 GridTemplate 的正式请求结构。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：当前要求固定传入 3 个符号，后续需要扩展时再改正式契约。
 */
export type GridTemplateRequest = {
  template: "grid";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  background: BackgroundRequest;
  slots: Array<{frames: SymbolRequest[]}>;
  layout: GridLayoutRequest;
  motion: GridMotionRequest;
  lyrics?: LyricItem[];
  font_path?: string;
};

/**
 * 功能说明：定义 ScrollTemplate 的布局参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：visible_cell_count 表示一张完整静态图应占多少等分格子。
 */
export type ScrollLayoutRequest = {
  visible_cell_count: number;
};

/**
 * 功能说明：定义 ScrollTemplate 的运动参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：loop_beats 仅在 loop=true 时生效，用于定义单轮循环跨越的拍数。
 */
export type ScrollMotionRequest = {
  loop: boolean;
  loop_beats?: number;
};

/**
 * 功能说明：定义 ScrollTemplate 的正式请求结构。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：当前为三元素连续循环条带模板。
 */
export type ScrollTemplateRequest = {
  template: "scroll";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  background: BackgroundRequest;
  slots: Array<{frames: SymbolRequest[]}>;
  layout: ScrollLayoutRequest;
  motion: ScrollMotionRequest;
  lyrics?: LyricItem[];
  font_path?: string;
};

/**
 * 功能说明：定义转场中的一个场景（独立背景 + 独立符号）。
 */
export type TransitionSceneRequest = {
  background: BackgroundRequest;
  symbol: SymbolRequest;
};

/**
 * 功能说明：定义镜头推移类模板的运动参数。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：travel_px 控制镜头移动的像素距离；easing 控制缓动曲线类型。
 */
export type PanMotionRequest = {
  travel_px: number;
  easing: "ease_in_out" | "ease_out" | "ease_in";
};

/**
 * 功能说明：定义 TiltUpTemplate 的正式请求结构（镜头上移，地面→天空）。
 * 镜头向上移动 → 旧场景下移出画、新场景从上方进入。
 */
export type TiltUpTemplateRequest = {
  template: "tilt_up";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  scene_before: TransitionSceneRequest;
  scene_after: TransitionSceneRequest;
  motion: PanMotionRequest;
  frames?: SymbolRequest[];
  energy_level?: "low" | "mid" | "high";
  rhythm_tension?: number;
  lyrics?: LyricItem[];
  font_path?: string;
};

/**
 * 功能说明：定义 TiltDownTemplate 的正式请求结构（镜头下移，天空→地面）。
 * 镜头向下移动 → 旧场景上移出画、新场景从下方进入。
 */
export type TiltDownTemplateRequest = {
  template: "tilt_down";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  scene_before: TransitionSceneRequest;
  scene_after: TransitionSceneRequest;
  motion: PanMotionRequest;
  frames?: SymbolRequest[];
  energy_level?: "low" | "mid" | "high";
  rhythm_tension?: number;
  lyrics?: LyricItem[];
  font_path?: string;
};

/**
 * 功能说明：定义 PanRightTemplate 的正式请求结构（镜头右移，"下一个"）。
 * 镜头向右移动 → 旧场景左移出画、新场景从右方进入。
 */
export type PanRightTemplateRequest = {
  template: "pan_right";
  fps: number;
  duration_in_frames: number;
  bpm: number;
  scene_before: TransitionSceneRequest;
  scene_after: TransitionSceneRequest;
  motion: PanMotionRequest;
  frames?: SymbolRequest[];
  energy_level?: "low" | "mid" | "high";
  rhythm_tension?: number;
  lyrics?: LyricItem[];
  font_path?: string;
};
