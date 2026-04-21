import os
import json
import argparse
from pathlib import Path

def calculate_clip_batch(task_id, runs_root="/root/data/runs", model_name="/root/data/t1/models/eval/clip/clip-vit-b32-laion2b"):
    """
    自动根据 task_id 匹配图片和 Prompt，计算 CLIP Score 并生成报告。
    """
    # 延迟加载重型库
    print("正在初始化深度学习环境 (加载 Torch/CLIP)...")
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm
    task_path = Path(runs_root) / task_id
    units_dir = task_path / "artifacts" / "module_b_units"
    frames_dir = task_path / "artifacts" / "frames"
    
    if not units_dir.exists() or not frames_dir.exists():
        print(f"错误: 找不到任务 {task_id} 的 artifacts 目录")
        return None

    # 1. 自动构建映射关系 (内存中匹配)
    prompt_mapping = {}
    unit_files = sorted(list(units_dir.glob("segment_*_seg_*.json")))
    
    print(f"正在扫描任务 {task_id} 的分镜数据...")
    for unit_file in unit_files:
        try:
            with open(unit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shot_id = data.get("shot_id")
                prompt_en = data.get("keyframe_prompt")
                prompt_zh = data.get("keyframe_prompt_zh") or "无中文提示词"
                if shot_id and prompt_en:
                    index_str = shot_id.split('_')[-1]
                    frame_name = f"frame_{index_str}.png"
                    if (frames_dir / frame_name).exists():
                        prompt_mapping[frame_name] = {
                            "en": prompt_en,
                            "zh": prompt_zh
                        }
        except Exception as e:
            print(f"解析 {unit_file.name} 出错: {e}")

    if not prompt_mapping:
        print("错误: 未找到可匹配的图片和 Prompt 映射")
        return None

    print(f"成功匹配 {len(prompt_mapping)} 条数据。")

    # 2. 计算评分
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型 {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    results = []
    for img_name, prompts in tqdm(prompt_mapping.items(), desc="计算得分"):
        img_full_path = frames_dir / img_name
        try:
            image = Image.open(img_full_path)
            inputs = processor(text=[prompts["en"]], images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            score = float(outputs.logits_per_image[0][0].item())
            results.append({
                "image": img_name,
                "prompt": prompts["en"],
                "prompt_zh": prompts["zh"],
                "clip_score": round(score, 4)
            })
        except Exception as e:
            print(f"处理 {img_name} 出错: {e}")
            
    results.sort(key=lambda x: x["clip_score"], reverse=True)
    return results

def main():
    print("\n=== CLIP Score 批量评测工具 (任务模式) ===")
    
    # 交互式输入
    task_id = input("请输入 Task ID (例如 jieranduhuo01): ").strip()
    if not task_id:
        print("错误: Task ID 不能为空")
        return

    runs_root = input("请输入 Runs 根目录 [默认 /root/data/runs]: ").strip() or "/root/data/runs"
    
    print(f"\n开始评测任务: {task_id}")
    report = calculate_clip_batch(task_id, runs_root)
    
    if report:
        output_path = Path(runs_root) / task_id / "clip_eval_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        print(f"\n评测完成！")
        print(f"结果报告: {output_path}")
        avg_score = sum(r['clip_score'] for r in report) / len(report)
        print(f"统计信息: 平均分 {avg_score:.4f} | 最高分 {report[0]['clip_score']} | 最低分 {report[-1]['clip_score']}")
    else:
        print("\n评测未完成。")

if __name__ == "__main__":
    main()
