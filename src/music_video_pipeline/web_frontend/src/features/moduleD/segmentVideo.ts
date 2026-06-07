/**
 * 文件用途：模块 D segment 合成视频路径解析与轻量存在性探测。
 */

/** 从 seg_XXXX 解析 segments 目录下的标准 mp4 相对路径。 */
export function resolveSegmentVideoBasePath(taskId: string, segmentId: string): string {
  const matched = segmentId.match(/(\d+)/);
  if (!matched) {
    return "";
  }
  const segNum = String(Number.parseInt(matched[1], 10)).padStart(3, "0");
  return `/task/${encodeURIComponent(taskId)}/artifacts/segments/segment_${segNum}.mp4`;
}

/** 拼接带缓存破除参数的视频 URL。 */
export function buildSegmentVideoUrl(basePath: string, cacheToken: number): string {
  if (!basePath) {
    return "";
  }
  return `${basePath}?t=${cacheToken}`;
}

/**
 * 用 Range 请求探测视频文件是否存在（只取 1 字节，避免整文件下载）。
 */
export async function probeSegmentVideoExists(basePath: string): Promise<boolean> {
  if (!basePath) {
    return false;
  }
  const probeUrl = `${basePath}?t=${Date.now()}`;
  try {
    const response = await fetch(probeUrl, {
      method: "GET",
      cache: "no-store",
      headers: {
        Range: "bytes=0-0",
      },
    });
    return response.status === 200 || response.status === 206;
  } catch {
    return false;
  }
}
