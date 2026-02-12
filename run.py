import os
import sys
import argparse
# 【关键】必须放在最前面导入
from omni.isaac.kit import SimulationApp

# 1. 在导入任何其他项目代码前，先配置并启动 Isaac Sim
# headless=False 表示显示图形界面，生成数据时如果为了速度可以改为 True
CONFIG = {
    "headless": False, 
    "width": 1280, 
    "height": 720
} 
simulation_app = SimulationApp(CONFIG)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 如果该路径不在 sys.path 中，则加入
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f"[INFO] Project Root added to sys.path: {PROJECT_ROOT}")

try:
    from scene.generator import SequenceGenerator
    from sim_utils.seed import set_seed  # 确保这里和实际文件夹名一致
except ImportError as e:
    print(f"[ERROR] Import failed. Sys.path is: {sys.path}")
    raise e

import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "config/dataset.yaml")
    cfg = load_config(config_path)

    # 固定随机种子
    set_seed(cfg['seed_base'])

    print(f"[INFO] Starting generation with {cfg['frames_per_scene']} frames per scene...")

    # 创建生成器
    generator = SequenceGenerator(
        config=cfg,
        shape_library_dir=os.path.join(PROJECT_ROOT, "models/isaac_models/Isaac/Props/Mugs")
    )

    # 批量生成
    try:
        generator.generate_sequences()
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭仿真
        simulation_app.close()