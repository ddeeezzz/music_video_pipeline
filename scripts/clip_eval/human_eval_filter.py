import os
import json
import shutil
from pathlib import Path

def prepare_human_eval(task_id, runs_root="/root/data/runs"):
    task_path = Path(runs_root) / task_id
    report_path = task_path / "clip_eval_report.json"
    frames_dir = task_path / "artifacts" / "frames"
    output_dir = task_path / "human_eval_samples"

    if not report_path.exists():
        print(f"错误: 找不到评测报告 {report_path}，请先执行 CLIP Score 评测。")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    if not report:
        print("错误: 报告内容为空。")
        return

    # 创建/清空输出目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 筛选逻辑：
    # 1. 最高分 3 张
    # 2. 最低分 3 张
    # 3. 中间分 (中位数附近) 3 张
    # 4. 随机抽样 3 张 (如果总数够多)
    
    samples = []
    count = len(report)
    
    # Top 3
    samples.extend([("TOP", r) for r in report[:3]])
    # Bottom 3
    samples.extend([("BOTTOM", r) for r in report[-3:]])
    # Middle 3
    mid = count // 2
    samples.extend([("MID", r) for r in report[max(0, mid-1):min(count, mid+2)]])

    print(f"\n=== 正在准备人工抽测样本 ({task_id}) ===")
    print(f"样本保存至: {output_dir}\n")

    summary_file = output_dir / "samples_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f_sum:
        f_sum.write(f"Task ID: {task_id} 人工抽测样本清单\n")
        f_sum.write("="*50 + "\n\n")

        for tag, item in samples:
            src = frames_dir / item['image']
            dst_name = f"{tag}_{item['clip_score']}_{item['image']}"
            dst = output_dir / dst_name
            
            if src.exists():
                shutil.copy2(src, dst)
                
                # 尝试从原始 unit JSON 文件中获取中文提示词（如果报告中没有）
                prompt_zh = item.get('prompt_zh')
                if not prompt_zh or prompt_zh == "无中文提示词":
                    # 根据图片名推断 unit 文件名，例如 frame_011.png -> segment_*_seg_0011.json
                    try:
                        frame_idx = item['image'].split('_')[-1].split('.')[0] # "0011"
                        unit_matches = list(units_dir.glob(f"segment_*_seg_{frame_idx}.json"))
                        if unit_matches:
                            with open(unit_matches[0], 'r', encoding='utf-8') as f_unit:
                                unit_data = json.load(f_unit)
                                prompt_zh = unit_data.get("keyframe_prompt_zh") or unit_data.get("keyframe_prompt_zh_v2")
                    except Exception:
                        pass
                
                prompt_zh = prompt_zh or "无中文提示词"
                log_line = f"[{tag}] Score: {item['clip_score']} | File: {dst_name}\n    [EN]: {item['prompt']}\n    [ZH]: {prompt_zh}\n"
                print(log_line.strip())
                f_sum.write(log_line + "-"*20 + "\n")
            else:
                print(f"警告: 找不到源文件 {src}")

    print(f"\n配置完成！共提取 {len(samples)} 个样本。")
    print(f"步骤提示：")
    print(f"1. 进入目录: {output_dir}")
    print(f"2. 重点对比 TOP 和 BOTTOM 组，观察 CLIP 分数是否符合你的视觉直观感受。")
    print(f"3. 检查 MID 组是否存在严重的线条噪点，判断当前 LoRA 权重的稳定性。")

if __name__ == "__main__":
    import sys
    t_id = sys.argv[1] if len(sys.argv) > 1 else input("请输入 Task ID 进行人工抽测准备: ").strip()
    if t_id:
        prepare_human_eval(t_id)
