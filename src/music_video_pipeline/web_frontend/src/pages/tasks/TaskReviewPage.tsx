import { useEffect, useMemo, useRef } from "react";

import {
  Alert,
  Button,
  Card,
  Empty,
  Image,
  Modal,
  Space,
  Typography,
} from "antd";
import { useQuery } from "@tanstack/react-query";

import { getTaskWebData, taskQueryKeys } from "@/api/taskApi";
import { appLogger } from "@/app/logger";
import { TaskStatusTag } from "@/features/tasks/components/TaskStatusTag";
import { useTaskIdParam } from "@/hooks/useTaskIdParam";
import { useReviewStore } from "@/stores/reviewStore";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) {
    return "00:00";
  }
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainSeconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainSeconds).padStart(2, "0")}`;
}

const CAMERA_MODE_ZH: Record<string, string> = {
  pan: "摇镜",
  zoom: "变焦",
  none: "固定",
};
const CAMERA_DIRECTION_ZH: Record<string, string> = {
  center: "居中",
  left: "左",
  right: "右",
  up: "上",
  down: "下",
  up_left: "左上",
  up_right: "右上",
  down_left: "左下",
  down_right: "右下",
};
const CAMERA_STRENGTH_ZH: Record<string, string> = {
  none: "无",
  small: "小幅",
  medium: "中幅",
};
const CAMERA_EASING_ZH: Record<string, string> = {
  linear: "线性",
  ease_in: "缓入",
  ease_out: "缓出",
  ease_in_out: "缓入缓出",
};

function formatCameraPlan(plan: Record<string, unknown> | undefined): string {
  if (!plan || !plan.mode) {
    return "无";
  }
  const mode = CAMERA_MODE_ZH[String(plan.mode)] || String(plan.mode);
  const direction = CAMERA_DIRECTION_ZH[String(plan.direction)] || String(plan.direction || "");
  const strength = CAMERA_STRENGTH_ZH[String(plan.strength)] || String(plan.strength || "");
  const easing = CAMERA_EASING_ZH[String(plan.easing)] || String(plan.easing || "");
  const presetId = String(plan.preset_id || "");
  return `${mode} ${direction} · ${strength} · ${easing}  (${presetId})`;
}

function scrollItemIntoContainer(container: HTMLDivElement | null, item: HTMLButtonElement | null): void {
  if (container === null || item === null) {
    return;
  }
  if (container.clientHeight <= 0) {
    return;
  }
  const nextScrollTop = item.offsetTop - container.clientHeight / 2 + item.clientHeight / 2;
  container.scrollTo({
    top: Math.max(0, nextScrollTop),
    behavior: "smooth",
  });
}

type TimelineItem = {
  start_time: number;
  end_time: number;
};

function findActiveTimelineItem<T extends TimelineItem>(items: T[], currentTime: number): T | null {
  if (!Number.isFinite(currentTime) || items.length === 0) {
    return null;
  }
  const lastIndex = items.length - 1;
  for (const [index, item] of items.entries()) {
    const startTime = Number(item.start_time);
    const endTime = Math.max(startTime, Number(item.end_time));
    if (!Number.isFinite(startTime) || !Number.isFinite(endTime)) {
      continue;
    }
    const isLastItem = index === lastIndex;
    if (currentTime >= startTime && (currentTime < endTime || (isLastItem && currentTime <= endTime))) {
      return item;
    }
  }
  return null;
}

export function TaskReviewPage() {
  const taskId = useTaskIdParam();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const segmentListRef = useRef<HTMLDivElement | null>(null);
  const lyricListRef = useRef<HTMLDivElement | null>(null);
  const segmentItemRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const lyricItemRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const currentTime = useReviewStore((state) => state.currentTime);
  const duration = useReviewStore((state) => state.duration);
  const selectedSegmentId = useReviewStore((state) => state.selectedSegmentId);
  const promptModalSegmentId = useReviewStore((state) => state.promptModalSegmentId);
  const setPlaybackState = useReviewStore((state) => state.setPlaybackState);
  const setSelectedSegmentId = useReviewStore((state) => state.setSelectedSegmentId);
  const openPromptModal = useReviewStore((state) => state.openPromptModal);
  const closePromptModal = useReviewStore((state) => state.closePromptModal);

  useEffect(() => {
    appLogger.info("音画审阅", "音画审阅页已进入", { taskId });
  }, [taskId]);

  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: taskQueryKeys.webData(taskId),
    queryFn: () => getTaskWebData(taskId),
    enabled: Boolean(taskId),
  });

  const activeSegment = useMemo(() => {
    return findActiveTimelineItem(data?.segment_units || [], currentTime);
  }, [currentTime, data?.segment_units]);

  const selectedSegment = useMemo(() => {
    const segments = data?.segment_units || [];
    return (
      segments.find((segment) => segment.segment_id === promptModalSegmentId) ||
      activeSegment ||
      segments.find((segment) => segment.segment_id === selectedSegmentId) ||
      null
    );
  }, [activeSegment, data?.segment_units, promptModalSegmentId, selectedSegmentId]);

  const activeLyricSegmentId = useMemo(() => {
    const lyric = findActiveTimelineItem(data?.lyric_units || [], currentTime);
    return lyric?.segment_id || "";
  }, [currentTime, data?.lyric_units]);

  useEffect(() => {
    if (!activeSegment?.segment_id) {
      return;
    }
    scrollItemIntoContainer(
      segmentListRef.current,
      segmentItemRefs.current[activeSegment.segment_id] || null,
    );
  }, [activeSegment?.segment_id]);

  useEffect(() => {
    if (!activeLyricSegmentId) {
      return;
    }
    scrollItemIntoContainer(
      lyricListRef.current,
      lyricItemRefs.current[activeLyricSegmentId] || null,
    );
  }, [activeLyricSegmentId]);

  return (
    <div className="page-stack review-page">
      {data?.video.available ? null : (
        <Alert
          type="warning"
          showIcon
          message="当前任务还没有可播放视频"
          description="如果任务还没跑到视频产物阶段，这里会先显示时间轴和文字信息。"
        />
      )}

      <div className="review-workspace">
        <Card
          bordered={false}
          loading={isLoading}
          className="review-panel review-panel--main"
          title="音画审阅"
          extra={(
            <Space wrap>
              <Button loading={isFetching && !isLoading} onClick={() => void refetch()}>
                刷新数据
              </Button>
              <TaskStatusTag status={data?.task_status || "unknown"} />
            </Space>
          )}
        >
          <div className="review-panel__body">
            <div className="review-player-section">
              <div className="review-video-shell">
                {data?.video.available ? (
                  <video
                    ref={videoRef}
                    className="review-video"
                    controls
                    src={data.video.url}
                    onLoadedMetadata={(event) => {
                      const element = event.currentTarget;
                      setPlaybackState(element.currentTime || 0, element.duration || 0);
                    }}
                    onTimeUpdate={(event) => {
                      const element = event.currentTarget;
                      setPlaybackState(element.currentTime || 0, element.duration || 0);
                    }}
                  />
                ) : (
                  <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="视频产物还没准备好" />
                )}
              </div>
              <div className="review-timebar">
                <Typography.Text>
                  当前时间：{formatTime(currentTime)} / {formatTime(duration)}
                </Typography.Text>
                {activeSegment ? (
                  <Typography.Text type="secondary">
                    当前 segment：{activeSegment.segment_id}
                  </Typography.Text>
                ) : null}
              </div>
            </div>
            <div className="review-focus">
              {selectedSegment ? (
                <>
                  <div className="review-focus__head">
                    <div className="review-focus__title-wrap">
                      <Typography.Title level={4} className="review-focus__title">
                        {selectedSegment.segment_id}
                      </Typography.Title>
                      <Space size={12} wrap className="review-focus__meta">
                        {selectedSegment.shot_id ? (
                          <Typography.Text type="secondary">shot：{selectedSegment.shot_id}</Typography.Text>
                        ) : null}
                        <Typography.Text type="secondary">
                          时间：{formatTime(selectedSegment.start_time)} - {formatTime(selectedSegment.end_time)}
                        </Typography.Text>
                      </Space>
                    </div>
                    <Button size="small" onClick={() => openPromptModal(selectedSegment.segment_id)}>
                      查看 prompt
                    </Button>
                  </div>
                  <Typography.Paragraph className="review-focus__desc">
                    {selectedSegment.scene_desc || "当前还没有场景描述。"}
                  </Typography.Paragraph>
                  <Typography.Text type="secondary">
                    运镜：{formatCameraPlan(selectedSegment.camera_plan as Record<string, unknown> | undefined)}
                  </Typography.Text>
                </>
              ) : (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="等待选中一个 segment" />
              )}
            </div>
          </div>
        </Card>

        <div className="review-side-column">
          <Card bordered={false} title="Segment 时间轴" className="scroll-card review-scroll-card">
            <div ref={segmentListRef} className="scroll-list">
              {(data?.segment_units || []).map((segment) => {
                const isActive = activeSegment?.segment_id === segment.segment_id;
                return (
                  <button
                    key={`${segment.segment_id}-${segment.start_time}`}
                    ref={(node) => {
                      segmentItemRefs.current[segment.segment_id] = node;
                    }}
                    type="button"
                    className={`scroll-item ${isActive ? "is-active" : ""}`}
                    onClick={() => {
                      setSelectedSegmentId(segment.segment_id);
                      if (videoRef.current) {
                        videoRef.current.currentTime = segment.start_time;
                      }
                      setPlaybackState(segment.start_time, videoRef.current?.duration || duration);
                    }}
                  >
                    <div className="scroll-item__head">
                      <Typography.Text strong>{segment.segment_id || "未命名 segment"}</Typography.Text>
                      <Typography.Text type="secondary">
                        {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                      </Typography.Text>
                    </div>
                    <Typography.Paragraph className="scroll-item__desc">
                      {segment.scene_desc || "当前还没有场景描述。"}
                    </Typography.Paragraph>
                    <Space wrap>
                      {segment.shot_id ? <Typography.Text type="secondary">shot: {segment.shot_id}</Typography.Text> : null}
                      <Button
                        size="small"
                        onClick={(event) => {
                          event.stopPropagation();
                          openPromptModal(segment.segment_id);
                        }}
                      >
                        查看 prompt
                      </Button>
                    </Space>
                  </button>
                );
              })}
            </div>
          </Card>

          <Card bordered={false} title="歌词时间轴" className="scroll-card review-scroll-card">
            <div ref={lyricListRef} className="scroll-list">
              {(data?.lyric_units || []).map((lyric, index) => {
                const isActive = activeLyricSegmentId === lyric.segment_id;
                const lyricRefKey = isActive ? activeLyricSegmentId : `${lyric.segment_id}-${index}`;
                return (
                  <button
                    key={`${lyric.segment_id}-${index}-${lyric.start_time}`}
                    ref={(node) => {
                      lyricItemRefs.current[lyricRefKey] = node;
                    }}
                    type="button"
                    className={`scroll-item ${isActive ? "is-active" : ""}`}
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = lyric.start_time;
                      }
                      setPlaybackState(lyric.start_time, videoRef.current?.duration || duration);
                    }}
                  >
                    <div className="scroll-item__head">
                      <Typography.Text>{lyric.text || "无歌词文本"}</Typography.Text>
                      <Typography.Text type="secondary">
                        {formatTime(lyric.start_time)}
                      </Typography.Text>
                    </div>
                  </button>
                );
              })}
            </div>
          </Card>
        </div>
      </div>

      <Modal
        title={selectedSegment ? `Prompt 详情：${selectedSegment.segment_id}` : "Prompt 详情"}
        open={Boolean(promptModalSegmentId)}
        onCancel={closePromptModal}
        footer={null}
        width={1160}
        centered
      >
        {selectedSegment ? (
          <div className="prompt-modal-grid">
            {/* 第一行：首尾帧图片 */}
            <div className="prompt-modal-row">
              <div className="prompt-modal-card prompt-modal-card--image">
                <div className="prompt-modal-card__label">首帧</div>
                {selectedSegment.frame_url_start ? (
                  <Image
                    className="prompt-modal-image"
                    src={selectedSegment.frame_url_start}
                    alt="首帧"
                    preview={false}
                  />
                ) : (
                  <div className="prompt-modal-empty">暂无首帧图片</div>
                )}
              </div>
              <div className="prompt-modal-card prompt-modal-card--image">
                <div className="prompt-modal-card__label">尾帧</div>
                {selectedSegment.frame_url_end ? (
                  <Image
                    className="prompt-modal-image"
                    src={selectedSegment.frame_url_end}
                    alt="尾帧"
                    preview={false}
                  />
                ) : (
                  <div className="prompt-modal-empty">暂无尾帧图片</div>
                )}
              </div>
            </div>

            {/* 第二行：首尾帧中文 prompt */}
            <div className="prompt-modal-row">
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">首帧中文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.keyframe_prompt_start_zh || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">尾帧中文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.keyframe_prompt_end_zh || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
            </div>

            {/* 第三行：首尾帧英文 prompt */}
            <div className="prompt-modal-row">
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">首帧英文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.keyframe_prompt_start_en || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">尾帧英文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.keyframe_prompt_end_en || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
            </div>

            {/* 第四行：视频 prompt */}
            <div className="prompt-modal-row">
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">视频中文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.video_prompt_zh || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
              <div className="prompt-modal-card prompt-modal-card--text">
                <div className="prompt-modal-card__label">视频英文 prompt</div>
                <div className="prompt-modal-card__body">
                  <Typography.Paragraph className="prompt-modal-card__text">
                    {selectedSegment.video_prompt_en || "-"}
                  </Typography.Paragraph>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
