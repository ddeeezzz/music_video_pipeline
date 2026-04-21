import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
from pathlib import Path

def calculate_clip_score(image_path, text_prompt, model_name="/root/data/t1/models/eval/clip/clip-vit-b32-laion2b"):
    """
    计算单张图片与对应文本的 CLIP Score。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和处理器
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 预处理输入
    image = Image.open(image_path)
    inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 获取相似度得分 (logits_per_image 是相似度矩阵)
    # 取值范围通常在 0-100 之间，越高越相关
    score = outputs.logits_per_image.item()
    
    return score

def main():
    parser = argparse.ArgumentParser(description="快速计算 CLIP Score 脚本")
    parser.add_argument("--image", type=str, required=True, help="待评测的图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="对应的文本提示词")
    parser.add_argument("--model", type=str, default="/root/data/t1/models/eval/clip/clip-vit-b32-laion2b", help="CLIP 模型名称")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 找不到图片文件 {args.image}")
        return

    print(f"正在计算 CLIP Score...")
    print(f"图片: {args.image}")
    print(f"Prompt: {args.prompt}")
    
    try:
        score = calculate_clip_score(args.image, args.prompt, args.model)
        print("-" * 30)
        print(f"CLIP Score: {score:.4f}")
        print("-" * 30)
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    main()
