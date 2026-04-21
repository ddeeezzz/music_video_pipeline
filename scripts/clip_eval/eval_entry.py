import os
import subprocess
import sys

def main():
    print("\n" + "="*30)
    print("      评测与筛选中心")
    print("="*30)
    print("[1] CLIP Score (自动跑分)")
    print("[2] 人工抽测 (样本筛选)")
    print("[3] 退出")
    print("-"*30)
    
    choice = input("请选择操作序号: ").strip()
    
    if choice == "1":
        # 运行批量评测脚本
        subprocess.run(["python3", "scripts/clip_eval/batch_clip_score.py"])
    elif choice == "2":
        # 运行人工抽测筛选脚本
        subprocess.run(["python3", "scripts/clip_eval/human_eval_filter.py"])
    elif choice == "3":
        print("已退出。")
    else:
        print("无效选项。")

if __name__ == "__main__":
    main()
