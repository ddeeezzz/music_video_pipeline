import os
import json
from pathlib import Path

def prepare_mapping(task_id, runs_root="/root/data/runs"):
    task_path = Path(runs_root) / task_id
    units_dir = task_path / "artifacts" / "module_b_units"
    frames_dir = task_path / "artifacts" / "frames"
    
    if not units_dir.exists() or not frames_dir.exists():
        print(f"错误: 找不到任务 {task_id} 的 artifacts 目录")
        return
    
    mapping = {}
    
    # 遍历所有的 unit JSON 文件
    unit_files = sorted(list(units_dir.glob("segment_*_seg_*.json")))
    
    for unit_file in unit_files:
        try:
            with open(unit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shot_id = data.get("shot_id")
                # 假设 keyframe_prompt 是评测的主要文本
                prompt = data.get("keyframe_prompt")
                
                if shot_id and prompt:
                    # 尝试匹配对应的帧文件，格式通常为 frame_001.png, frame_002.png...
                    # shot_id 格式通常为 shot_001, shot_002...
                    index_str = shot_id.split('_')[-1] # "001"
                    frame_name = f"frame_{index_str}.png"
                    
                    if (frames_dir / frame_name).exists():
                        mapping[frame_name] = prompt
                    else:
                        print(f"警告: 找不到对应的帧文件 {frame_name} (来自 {unit_file.name})")
        except Exception as e:
            print(f"解析 {unit_file.name} 出错: {e}")
            
    output_path = task_path / "clip_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    
    print(f"成功！已为任务 {task_id} 准备好 mapping 文件:")
    print(f"路径: {output_path}")
    print(f"总计条目: {len(mapping)}")
    return output_path

if __name__ == "__main__":
    prepare_mapping("jieranduhuo01")
