/**
 * 文件用途：渲染歌词叠加层，支持原文+翻译双行显示。
 * 核心流程：根据当前帧匹配歌词项 -> 渲染原文（+翻译），白字黑描边。
 * 输入输出：输入歌词数组、当前帧，输出歌词 JSX 叠加层。
 * 依赖说明：依赖 remotion useCurrentFrame、react。
 * 维护说明：字体通过 staticFile 加载 public/fonts/ 中的 NotoSansCJKsc 字体。
 */

// 第三方库：用于读取当前帧和静态文件路径。
import {staticFile, useCurrentFrame} from "remotion";
// 第三方库：用于声明组件返回类型。
import type {ReactElement} from "react";
// 项目内模块：用于歌词类型。
import type {LyricItem} from "../types";

// 常量：歌词渲染区域宽度（内容不超出 80% 画布）。
const LYRICS_MAX_WIDTH_PX = 1075; // 1344 * 0.8

/**
 * 功能说明：渲染当前帧对应的歌词（原文 + 可选翻译）。
 * 参数说明：
 * - lyrics: 歌词数组（已转换为帧坐标）。
 * 返回值：
 * - ReactElement | null: 有歌词时返回歌词 JSX，否则 null。
 * 异常说明：无。
 * 边界条件：歌词为空或当前帧无匹配时返回 null。
 */
export const LyricsOverlay = ({
  lyrics,
}: {
  lyrics?: LyricItem[];
}): ReactElement | null => {
  const frame = useCurrentFrame();

  if (!lyrics || lyrics.length === 0) {
    return null;
  }

  // 查找当前帧匹配的歌词
  const activeLyric = lyrics.find(
    (item) => frame >= item.start_frame && frame < item.end_frame,
  );

  if (!activeLyric) {
    return null;
  }

  const hasTranslation =
    activeLyric.translated_text &&
    activeLyric.translated_text.trim().length > 0;

  const baseStyle: Record<string, string | number> = {
    fontFamily: '"NotoSansCJKsc", sans-serif',
    fontWeight: 700,
    color: "#FFFFFF",
    textShadow:
      "0 0 2px #000, 0 0 2px #000, 0 0 2px #000, 0 0 2px #000, 0 0 2px #000, 0 0 3px #000",
    WebkitTextStroke: "1px #000",
    lineHeight: 1.5,
    padding: "4px 16px",
    display: "inline-block",
    borderRadius: 4,
  };

  return (
    <>
      <style>{`
        @font-face {
          font-family: 'NotoSansCJKsc';
          src: url('${staticFile("/fonts/NotoSansCJKsc-Regular.otf")}') format('opentype');
          font-weight: 700;
          font-style: normal;
        }
      `}</style>
      <div
        style={{
          position: "absolute",
          bottom: "8%",
          left: "50%",
          transform: "translateX(-50%)",
          width: LYRICS_MAX_WIDTH_PX,
          maxWidth: "80%",
          textAlign: "center",
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <div style={{...baseStyle, fontSize: 36}}>
          {activeLyric.text}
        </div>
        {hasTranslation && (
          <div style={{...baseStyle, fontSize: 26, marginTop: 4}}>
            {activeLyric.translated_text}
          </div>
        )}
      </div>
    </>
  );
};
