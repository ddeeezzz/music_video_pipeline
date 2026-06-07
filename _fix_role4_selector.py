import re
path = r'm:\MyTest\working\music_video_pipeline\src\music_video_pipeline\monitoring\handlers\module_b.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start and end of _load_role4_shot_selector_items
start_marker = '    def _load_role4_shot_selector_items(self, task_dir: Path) -> list[dict[str, Any]]:'
end_marker = '    @staticmethod'

# Find the method
start_idx = content.find(start_marker)
if start_idx == -1:
    print("ERROR: method not found")
    exit(1)

# Find the next method (end of this method)
end_idx = content.find(end_marker, start_idx + len(start_marker))
if end_idx == -1:
    print("ERROR: end of method not found")
    exit(1)

print(f"Found method at {start_idx}-{end_idx}")

# Build replacement
new_method = '''    def _load_role4_shot_selector_items(self, task_dir: Path) -> list[dict[str, Any]]:
        """
        功能说明：从 role3 流式文件解析 shot 列表，作为 role4 的按 shot 操作选择项。
        参数说明：
        - task_dir: 任务目录。
        返回值：
        - list[dict[str, Any]]: shot 入口数组，每项含 segment_id/shot_id/scene_desc/big_segment_id。
        异常说明：无；缺失 role3 流式产物时回退为空数组。
        边界条件：逐个大段流式文件独立读取并解析。
        """

        # 从模块 A 输出读取 segment 时间
        segment_times: dict[str, tuple[float, float]] = {}
        module_a_path = task_dir / "artifacts" / "module_a_output.json"
        if module_a_path.exists():
            try:
                ma_payload = json.loads(module_a_path.read_text(encoding="utf-8"))
                for seg in (ma_payload.get("segments", []) if isinstance(ma_payload, dict) else []):
                    sid = str(seg.get("segment_id", "")).strip()
                    if sid:
                        st = float(seg.get("start_time", 0) or 0)
                        et = float(seg.get("end_time", st) or 0)
                        segment_times[sid] = (st, et)
            except Exception:
                pass

        streaming_dir = get_module_b_streaming_dir((task_dir / "artifacts").resolve(), "role3")
        items: list[dict[str, Any]] = []
        if not streaming_dir.exists():
            return items
        for stream_path in sorted(streaming_dir.glob("role3_segment_output.streaming.*.md")):
            try:
                text = stream_path.read_text(encoding="utf-8").replace("\\r\\n", "\\n")
            except Exception:
                continue
            # 从文件名提取 big_segment_id: role3_segment_output.streaming.big_001.md
            current_big = stream_path.stem.replace("role3_segment_output.streaming.", "").strip()

            for block in re.split(r"\\n(?=### )", text):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\\n")
                heading = lines[0].strip()
                if heading.startswith("## "):
                    current_big = heading[3:].strip()
                    continue
                if not heading.startswith("### "):
                    continue
                seg_id = heading[4:].strip()
                if not seg_id:
                    continue
                scene_desc = ""
                remotion_id = ""
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith("- scene_desc_zh:"):
                        scene_desc = stripped[len("- scene_desc_zh:"):].strip()
                    elif stripped.startswith("- remotion_id:"):
                        remotion_id = stripped[len("- remotion_id:"):].strip()

                subjects = _parse_subject_descriptions(scene_desc, remotion_id)
                start_time, end_time = segment_times.get(seg_id, (0.0, 0.0))

                for subj_idx, subject_desc in enumerate(subjects, start=1):
                    shot_id = _build_shot_id(seg_id, subj_idx)
                    label = f"{scene_desc} / {seg_id} / {current_big}"

                    items.append({
                        "segment_id": shot_id,
                        "shot_id": shot_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": label,
                        "role": "role4",
                        "scene_desc": scene_desc,
                        "big_segment_id": current_big,
                        "remotion_id": remotion_id,
                    })
        return items

    '''

old_method = content[start_idx:end_idx]
content = content.replace(old_method, new_method, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("OK")
