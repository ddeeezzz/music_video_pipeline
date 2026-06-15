/**
 * 文件用途：定义 Remotion Studio Props 面板所需的正式 schema。
 * 核心流程：复用模板请求的最小契约结构 -> 为 Center、Grid、Scroll 提供可编辑表单。
 * 输入输出：输入为 Studio 当前 props 值，输出为经 Zod 校验后的正式 props 结构。
 * 依赖说明：依赖 zod 作为 schema 定义库。
 * 维护说明：此处只承载模板交互编辑所需字段，不额外扩展为另一套配置系统。
 */

// 第三方库：用于定义 Remotion Composition 的 props schema。
import {z} from "zod";

// 常量：背景类型枚举。
const backgroundKindSchema = z.enum(["none", "solid", "image", "video"]);
// 常量：单个符号请求 schema。
const symbolRequestSchema = z.object({
  src: z.string().min(1),
  width_ratio: z.number().gt(0).lte(1),
  height_ratio: z.number().gt(0).lte(1)
});

/**
 * 功能说明：定义背景请求 schema。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：为简化 Studio 表单，统一保留 color/src 字段，由运行时按 kind 消费。
 */
export const backgroundRequestSchema = z.object({
  kind: backgroundKindSchema,
  color: z.string().optional(),
  src: z.string().optional()
});

/**
 * 功能说明：定义单行歌词字段 schema。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：start_frame/end_frame 为相对模板时间轴的帧序号。
 */
const lyricItemSchema = z.object({
  text: z.string(),
  translated_text: z.string().optional(),
  start_frame: z.number().int().min(0),
  end_frame: z.number().int().min(0),
});

/**
 * 功能说明：定义 CenterTemplate 的 props schema。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：画布宽高已固定，不再暴露为可编辑字段。
 */
export const centerTemplateSchema = z.object({
  template: z.literal("center"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  background: backgroundRequestSchema,
  frames: z.array(symbolRequestSchema),
  motion: z.object({
    breathe: z.boolean()
  }),
  energy_level: z.enum(["low", "mid", "high"]).optional(),
  rhythm_tension: z.number().min(0).max(1).optional(),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});

/**
 * 功能说明：定义 GridTemplate 的 props schema。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：当前固定要求 3 个符号。
 */
export const gridTemplateSchema = z.object({
  template: z.literal("grid"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  background: backgroundRequestSchema,
  slots: z.array(z.object({
    frames: z.array(symbolRequestSchema),
  })),
  layout: z.object({
    visible_cell_count: z.number().int().positive()
  }),
  motion: z.object({
    active_ratio: z.number().gt(0).lte(1),
    overshoot_ratio: z.number().min(0),
    enter_distance: z.number().min(0)
  }),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});

/**
 * 功能说明：定义 ScrollTemplate 的 props schema。
 * 参数说明：无。
 * 返回值：不适用。
 * 异常说明：不适用。
 * 边界条件：当前固定要求 3 个符号，循环节奏仅在 loop=true 时允许传入。
 */
export const scrollTemplateSchema = z.object({
  template: z.literal("scroll"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  background: backgroundRequestSchema,
  slots: z.array(z.object({
    frames: z.array(symbolRequestSchema),
  })),
  layout: z.object({
    visible_cell_count: z.number().int().positive()
  }),
  motion: z.object({
    loop: z.boolean(),
    loop_beats: z.number().positive().optional()
  }).superRefine((value, context) => {
    if (!value.loop && value.loop_beats !== undefined) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "loop=false 时不应传入 loop_beats。"
      });
    }
  }),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});

/**
 * 功能说明：定义转场场景的通用 schema（独立背景 + 独立符号）。
 */
const transitionSceneSchema = z.object({
  background: backgroundRequestSchema,
  symbol: symbolRequestSchema
});

/**
 * 功能说明：定义镜头推移类模板的通用 motion schema。
 */
const panMotionSchema = z.object({
  travel_px: z.number().min(0).default(80),
  easing: z.enum(["ease_in_out", "ease_out", "ease_in"]).default("ease_in_out")
});

/**
 * 功能说明：定义 TiltUpTemplate 的 props schema（镜头上移）。
 */
export const tiltUpTemplateSchema = z.object({
  template: z.literal("tilt_up"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  scene_before: transitionSceneSchema,
  scene_after: transitionSceneSchema,
  motion: panMotionSchema,
  frames: z.array(symbolRequestSchema).optional(),
  energy_level: z.enum(["low", "mid", "high"]).optional(),
  rhythm_tension: z.number().min(0).max(1).optional(),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});

/**
 * 功能说明：定义 TiltDownTemplate 的 props schema（镜头下移）。
 */
export const tiltDownTemplateSchema = z.object({
  template: z.literal("tilt_down"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  scene_before: transitionSceneSchema,
  scene_after: transitionSceneSchema,
  motion: panMotionSchema,
  frames: z.array(symbolRequestSchema).optional(),
  energy_level: z.enum(["low", "mid", "high"]).optional(),
  rhythm_tension: z.number().min(0).max(1).optional(),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});

/**
 * 功能说明：定义 PanRightTemplate 的 props schema（镜头右移）。
 */
export const panRightTemplateSchema = z.object({
  template: z.literal("pan_right"),
  fps: z.number().int().positive(),
  duration_in_frames: z.number().int().positive(),
  bpm: z.number().positive(),
  scene_before: transitionSceneSchema,
  scene_after: transitionSceneSchema,
  motion: panMotionSchema,
  frames: z.array(symbolRequestSchema).optional(),
  energy_level: z.enum(["low", "mid", "high"]).optional(),
  rhythm_tension: z.number().min(0).max(1).optional(),
  lyrics: z.array(lyricItemSchema).optional(),
  font_path: z.string().optional(),
});
