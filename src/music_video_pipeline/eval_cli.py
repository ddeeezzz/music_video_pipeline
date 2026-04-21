import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取项目根目录 (假设此文件在 src/music_video_pipeline/)
    project_root = Path(__file__).parent.parent.parent
    scripts_dir = project_root / "scripts" / "clip_eval"

    print("\n" + "="*40)
    print("      流水线评测与抽测系统")
    print("="*40)
    print("[1] CLIP Score (全量自动跑分)")
    print("[2] 人工抽测 (样本自动筛选)")
    print("[3] 退出")
    print("-"*40)
    
    choice = input("请选择操作序号 [1-3]: ").strip()
    
    if choice == "1":
        script_path = scripts_dir / "batch_clip_score.py"
        subprocess.run([sys.executable, str(script_path)])
    elif choice == "2":
        script_path = scripts_dir / "human_eval_filter.py"
        subprocess.run([sys.executable, str(script_path)])
    elif choice == "3":
        print("已退出评测系统。")
    else:
        print("无效选项，请重新运行。")

if __name__ == "__main__":
    main()
