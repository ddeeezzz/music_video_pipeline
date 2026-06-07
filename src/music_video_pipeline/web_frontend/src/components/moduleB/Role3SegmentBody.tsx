import { useMemo } from "react";
import { Button, Empty, Select, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import type { TaskModuleBRole } from "@/schemas/moduleB";

interface BigSegmentGroup {
  big_segment_id: string;
  story_outline_zh: string;
  display_title: string;
  display_subtitle: string;
  shots: any[];
  start_time: number;
  end_time: number;
}

interface Role3SegmentBodyProps {
  role: TaskModuleBRole;
  selectedBigSegmentId: string;
  onBigSegmentChange: (bid: string) => void;
  onSegmentRerun: (bid: string) => void;
  roleRerunLoading: boolean;
  segmentRerunLoading: boolean;
}

function formatTimeRange(startTime: number, endTime: number): string {
  return `${startTime.toFixed(2)} - ${endTime.toFixed(2)}`;
}

export function Role3SegmentBody({
  role,
  selectedBigSegmentId,
  onBigSegmentChange,
  onSegmentRerun,
  roleRerunLoading,
  segmentRerunLoading,
}: Role3SegmentBodyProps) {
  const roleSegments = role.segment_items || [];

  const bigSegmentGroups = useMemo(() => {
    const groups: Record<string, BigSegmentGroup> = {};
    for (const item of roleSegments) {
      const bid = (item as any).big_segment_id || item.segment_id;
      if (!groups[bid]) {
        groups[bid] = {
          big_segment_id: bid,
          story_outline_zh: (item as any).story_outline_zh || "",
          display_title: (item as any).display_title || bid,
          display_subtitle: (item as any).display_subtitle || "",
          shots: [],
          start_time: item.start_time,
          end_time: item.end_time,
        };
      }
      groups[bid].shots.push(item);
      groups[bid].start_time = Math.min(groups[bid].start_time, item.start_time);
      groups[bid].end_time = Math.max(groups[bid].end_time, item.end_time);
    }
    return Object.values(groups);
  }, [roleSegments]);

  const bigSegmentOptions = useMemo(
    () =>
      bigSegmentGroups.map((group) => ({
        value: group.big_segment_id,
        label: group.display_title || group.big_segment_id,
      })),
    [bigSegmentGroups],
  );

  const selectedGroup = bigSegmentGroups.find((g) => g.big_segment_id === selectedBigSegmentId);

  return (
    <div className="module-b-segment-box">
      <div className="module-b-segment-toolbar">
        <Select
          value={selectedBigSegmentId || null}
          placeholder="选择 big_segment"
          options={bigSegmentOptions}
          onChange={(value) => onBigSegmentChange(String(value))}
          className="module-b-segment-select"
        />
        <Button
          icon={<ReloadOutlined />}
          loading={segmentRerunLoading}
          onClick={() => {
            if (!selectedBigSegmentId) return;
            onSegmentRerun(selectedBigSegmentId);
          }}
        >
          按 Big Segment 重跑
        </Button>
      </div>
      {selectedGroup ? (
        <div className="module-b-segment-summary">
          <Typography.Text strong>{selectedGroup.big_segment_id}</Typography.Text>
          <Typography.Text type="secondary">
            {formatTimeRange(selectedGroup.start_time, selectedGroup.end_time)}
          </Typography.Text>
          <Typography.Text type="secondary">
            {selectedGroup.display_subtitle || "未标注"}
          </Typography.Text>
          <pre className="module-b-streaming-content">
            {role.stream_preview_segments?.find(
              (s) => s.segment_id === selectedBigSegmentId,
            )?.content || "…"}
          </pre>
        </div>
      ) : (
        <Empty description="当前任务还没有可用于该角色的 segment 列表。" />
      )}
    </div>
  );
}
