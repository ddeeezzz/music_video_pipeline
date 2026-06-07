import { useMemo } from "react";
import { Alert, Button, Empty, Select, Space, Tag, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import type { TaskModuleBRole } from "@/schemas/moduleB";

interface Role4SegmentBodyProps {
  role: TaskModuleBRole;
  selectedSegmentId: string;
  onSegmentChange: (sid: string) => void;
  onSegmentRerun: (sid: string) => void;
  roleRerunLoading: boolean;
  segmentRerunLoading: boolean;
}

function formatTimeRange(startTime: number, endTime: number): string {
  return `${startTime.toFixed(2)} - ${endTime.toFixed(2)}`;
}

function formatSegmentOptionLabel(
  segmentId: string,
  shotId: string,
  label: string,
  displayTitle?: string,
): string {
  const normalizedDisplayTitle = (displayTitle || "").trim();
  if (normalizedDisplayTitle) {
    return normalizedDisplayTitle;
  }
  const normalizedLabel = label.trim();
  const idPart = segmentId === shotId ? segmentId : `${segmentId} / ${shotId}`;
  if (!normalizedLabel) {
    return idPart;
  }
  return `${idPart} / ${normalizedLabel}`;
}

export function Role4SegmentBody({
  role,
  selectedSegmentId,
  onSegmentChange,
  onSegmentRerun,
  roleRerunLoading,
  segmentRerunLoading,
}: Role4SegmentBodyProps) {
  const roleSegments = role.segment_items || [];
  const selectedSegment = roleSegments.find((item) => item.segment_id === selectedSegmentId);

  const shotOptions = useMemo(
    () =>
      roleSegments.map((item) => ({
        value: item.segment_id,
        label: formatSegmentOptionLabel(
          item.segment_id,
          item.shot_id,
          item.label,
          (item as any).display_title,
        ),
      })),
    [roleSegments],
  );

  return (
    <div className="module-b-segment-box">
      <div className="module-b-segment-toolbar">
        <Select
          value={selectedSegmentId || null}
          placeholder="选择 shot"
          options={shotOptions}
          onChange={(value) => onSegmentChange(String(value))}
          className="module-b-segment-select"
        />
        <Button
          icon={<ReloadOutlined />}
          loading={segmentRerunLoading}
          onClick={() => {
            if (!selectedSegmentId) return;
            onSegmentRerun(selectedSegmentId);
          }}
        >
          按 Shot 重跑
        </Button>
      </div>
      {selectedSegment ? (
        <div className="module-b-segment-summary">
          <Typography.Text strong>
            {(selectedSegment as any).display_shot_id || selectedSegment.shot_id || selectedSegment.segment_id}
          </Typography.Text>
          <Typography.Text type="secondary">
            {formatTimeRange(selectedSegment.start_time, selectedSegment.end_time)}
          </Typography.Text>
          <Typography.Text type="secondary">
            {(selectedSegment as any).display_subtitle || selectedSegment.label || selectedSegment.role || "未标注"}
          </Typography.Text>
          <pre className="module-b-streaming-content">
            {role.stream_preview_segments?.find(
              (s) => s.segment_id === selectedSegment.segment_id,
            )?.content || "…"}
          </pre>
        </div>
      ) : (
        <Empty description="当前任务还没有可用于该角色的 segment 列表。" />
      )}
    </div>
  );
}
