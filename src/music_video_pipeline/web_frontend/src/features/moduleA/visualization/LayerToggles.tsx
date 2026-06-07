import { Checkbox } from "antd";
import { useVisualizationStore, type LayerKey } from "@/stores/visualizationStore";

const LAYER_DEFS: { key: LayerKey; title: string }[] = [
  { key: "a0", title: "A0段（Allin1直出）" },
  { key: "al", title: "A1段（单次边界矫正）" },
  { key: "b", title: "B段（最终大段）" },
  { key: "s", title: "S段（最终小段）" },
  { key: "role", title: "ContentRoles" },
  { key: "beats", title: "Beats" },
  { key: "lyrics", title: "Lyrics（全量分句）" },
  { key: "lyrics_attached", title: "Lyrics（挂载版）" },
  { key: "energy", title: "Energy" },
  { key: "onset", title: "Onset" },
  { key: "precheck", title: "人声RMS预检" },
  { key: "accompaniment_rms", title: "伴奏RMS" },
];

export function LayerToggles() {
  const layerVisibility = useVisualizationStore((s) => s.layerVisibility);
  const toggleLayer = useVisualizationStore((s) => s.toggleLayer);

  const allChecked = Object.values(layerVisibility).every(Boolean);
  const someChecked = Object.values(layerVisibility).some(Boolean);

  return (
    <div>
      <div style={{ marginBottom: 4 }}>
        <Checkbox
          checked={allChecked}
          indeterminate={someChecked && !allChecked}
          onChange={() => {
            const newVal = !allChecked;
            for (const def of LAYER_DEFS) {
              useVisualizationStore.getState().setLayerVisibility(def.key, newVal);
            }
          }}
        >
          全选 / 取消
        </Checkbox>
      </div>
      <div className="vis-layer-grid">
        {LAYER_DEFS.map((def) => (
          <Checkbox
            key={def.key}
            checked={layerVisibility[def.key]}
            onChange={() => toggleLayer(def.key)}
          >
            {def.title}
          </Checkbox>
        ))}
      </div>
    </div>
  );
}
