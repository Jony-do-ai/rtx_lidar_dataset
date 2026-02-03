import os
import time
import numpy as np
import open3d as o3d
from PIL import Image

# Isaac Sim 核心模块
import omni.replicator.core as rep
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, DynamicCuboid, DynamicSphere
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import LidarRtx
# from omni.replicator.core.scripts.utils import get_prim_path
from pxr import Usd, UsdLux, Sdf
import omni.usd
from pxr import UsdLux
# 自定义模块 (请确保 rtx_lidar_dataset/data/writer.py 存在)
from file_io.writer import save_pointcloud


class SequenceGenerator:
    def __init__(self, config, shape_library_dir):
        self.cfg = config
        self.num_seq = config['num_scenes']
        self.frames_per_seq = config['frames_per_scene']
        self.output_dir = config['output_dir']

        # 统一 XYZ 坐标系：World 坐标系，单位米
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()


    def setup_scene(self):
        """搭建场景：确保所有组件在同一坐标系"""
        # 地面在 Z=0
        self.ground = GroundPlane(prim_path="/World/GroundPlane", size=100.0, color=np.array([0.5, 0.5, 0.5]))

        # 创建固定路径的光源，方便后续通过 USD 属性精准控制
        self.main_light = rep.create.light(
            light_type="distant",
            intensity=1500,
            name="MyMovingLight"
        )
        self.light_path = "/Replicator/MyMovingLight"

        # 固定相机视角：所有帧共享同一视点，确保差分对齐
        self.camera_rig = rep.create.camera(position=[0.0, -4.0, 2.5], look_at=[0.0, 0.0, 0.5])
        self.render_product = rep.create.render_product(self.camera_rig, self.cfg['sensor']['camera']['resolution'])

        self.world.reset()

    def set_light_shadows(self, enable: bool):
        """控制底层 USD 属性切换阴影"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.light_path)
        if prim.IsValid():
            # 针对不同版本的 Isaac Sim 尝试属性名
            for attr_name in ["inputs:shadow:enable", "inputs:enableShadows"]:
                attr = prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(bool(enable))
                    return True
        return False

    def generate_sequences(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # ----------------------------------------
        # 1. 注册 Annotators
        # ----------------------------------------
        rgb_annotator = rep.annotators.get("rgb")
        rgb_annotator.attach(self.render_product)

        object_prim_path = "/World/TargetObject"

        # 预热渲染器
        rep.orchestrator.step()

        for seq_idx in range(self.num_seq):
            seq_name = f"seq_{seq_idx:03d}"
            seq_dir = os.path.join(self.output_dir, "sequences", seq_name)
            geom_dir = os.path.join(seq_dir, "object_geometry")
            os.makedirs(geom_dir, exist_ok=True)

            print(f"[{seq_idx + 1}/{self.num_seq}] Generating {seq_name}...")

            # --- 0. 清理旧物体
            self.world.stop()
            if is_prim_path_valid(object_prim_path):
                if self.world.scene.object_exists("target_object"):
                    self.world.scene.remove_object("target_object")
                delete_prim(object_prim_path)
                omni.kit.app.get_app().update()

            # --- 1. 随机生成一个几何体并保持位置固定
            obj_scale = np.array([0.5, 0.5, 0.5])
            # 核心修正：Z轴位置 = 高度的一半
            obj_pos = np.array([0, 0, obj_scale[2] / 2.0])
            self.current_object = FixedCuboid(prim_path=object_prim_path,
                                                position=obj_pos,
                                                scale=obj_scale,
                                                color=np.array([1.0, 0.0, 0.0])  # 设置颜色为红色
                                                )
            self.world.reset()

            # 3. 预热渲染器确保缓存清空
            for _ in range(10):
                self.world.step(render=True)

            # 4. 抓取图像：这张图就是“纯几何体 + 纯白背景”且无阴影
            rep.orchestrator.step()
            base_rgb = rgb_annotator.get_data()[..., :3].copy()
            Image.fromarray(base_rgb).save(os.path.join(geom_dir, "base_no_shadow.png"))

            # 5. 结束后关闭环境光，准备生成带阴影的帧
            # ambient_light.visible = False

            # --- 3. 生成唯一的点云标注 (XYZ 统一坐标) ---
            # 直接从 Mesh 采样以获得完整、稳定的真值
            points = np.random.uniform(-0.25, 0.25, (5000, 3)) + np.array([0, 0, 0.5])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(geom_dir, "gt_fixed.ply"), pcd, write_ascii=True)

            # 预跑一步
            rep.orchestrator.step()

            for frame_idx in range(self.frames_per_seq):
                frame_dir = os.path.join(seq_dir, f"frame_{frame_idx:03d}")
                os.makedirs(frame_dir, exist_ok=True)

                phi = (frame_idx / self.frames_per_seq) * 360.0
                theta = 45.0

                # 开启阴影并旋转光线
                self.set_light_shadows(True)
                with self.main_light:
                    rep.modify.pose(rotation=(theta, 0, phi))

                rep.orchestrator.step()
                frame_rgb = rgb_annotator.get_data()[..., :3].copy()
                Image.fromarray(frame_rgb).save(os.path.join(frame_dir, "rgb_with_shadow.png"))

                # --- 5. 差分与掩码提取 ---
                diff = np.abs(base_rgb.astype(np.float32) - frame_rgb.astype(np.float32))
                diff_img = diff.mean(axis=2).astype(np.uint8)
                shadow_mask = (diff_img > 20).astype(np.uint8) * 255  # 阈值化提取阴影

                Image.fromarray(diff_img).save(os.path.join(frame_dir, "diff_visual.png"))
                Image.fromarray(shadow_mask).save(os.path.join(frame_dir, "shadow_mask.png"))

                # 保存光线参数供训练使用
                with open(os.path.join(frame_dir, "light_info.txt"), "w") as f:
                    f.write(f"theta:{theta}, phi:{phi}")