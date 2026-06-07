import type { ModuleAVisualizationPayload } from "@/schemas/moduleAVisualization";

function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const min = Math.floor(s / 60);
  const sec = s % 60;
  return `${min}m ${sec.toFixed(1)}s`;
}

export function SummaryCards({ payload }: { payload: ModuleAVisualizationPayload }) {
  const { summary, duration_seconds, task_id, vocal_precheck_rms } = payload;
  const bs = summary.boundary_shift;

  const cards = [
    {
      key: "task",
      title: "任务ID",
      value: task_id,
      sub: `总时长 ${formatSeconds(duration_seconds || 0)}`,
    },
    {
      key: "segments",
      title: "段落统计",
      value: `A0=${summary.a0_count}  A1=${summary.al_count}  B=${summary.b_count}  S=${summary.s_count}`,
      sub: "",
    },
    {
      key: "beats-lyric",
      title: "节拍与歌词",
      value: `Beats=${summary.beat_count}  Lyrics=${summary.lyric_count}  Energy=${summary.energy_count}`,
      sub: `挂载歌词=${summary.lyric_attached_count}`,
    },
    {
      key: "boundary",
      title: "A0→A1 边界调整",
      value: `${bs.adjusted_count}/${bs.compared_count} 段`,
      sub: `平均偏移 ${bs.average_abs_shift_seconds?.toFixed(3)}s  /  最大 ${bs.max_abs_shift_seconds?.toFixed(3)}s`,
    },
    {
      key: "precheck",
      title: "FunASR 预检",
      value: vocal_precheck_rms?.should_skip_funasr ? "已跳过" : "已通过",
      sub: `peak=${vocal_precheck_rms?.peak_rms?.toFixed(4) ?? "-"}  threshold=${vocal_precheck_rms?.peak_threshold?.toFixed(4) ?? "-"}`,
    },
  ];

  return (
    <div className="vis-cards">
      {cards.map((card) => (
        <div key={card.key} className="vis-card">
          <div className="vis-card__title">{card.title}</div>
          <div className="vis-card__value">{card.value}</div>
          {card.sub && <div className="vis-card__detail">{card.sub}</div>}
        </div>
      ))}
    </div>
  );
}
