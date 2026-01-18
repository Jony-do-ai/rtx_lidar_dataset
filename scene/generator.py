import os
import time
import numpy as np
import open3d as o3d
from PIL import Image

# Isaac Sim 核心模块
import omni.replicator.core as rep
import omni.syntheticdata as sd
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, DynamicCuboid, DynamicSphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera, LidarRtx
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid

# 自定义模块
from file_io.writer import save_pointcloud
from sim_utils.transform import mesh_to_pointcloud


class SequenceGenerator:
    def __init__(self, config, shape_library_dir):
        self.cfg = config
        self.num_seq = config['num_scenes']
        self.frames_per_seq = config['frames_per_scene']
        self.output_dir = config['output_dir']
        self.shape_library_dir = shape_library_dir

        # 初始化 World
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()

    def setup_scene(self):
        """搭建基础场景：地面、光照、传感器"""
        self.ground = GroundPlane(prim_path="/World/GroundPlane", size=100.0, color=np.array([0.5, 0.5, 0.5]))

        # 光照角度调整为 75 度，拉长阴影
        rep.create.light(light_type="distant", rotation=(75, 0, 0), intensity=1000)

        cam_cfg = self.cfg['sensor']['camera']
        self.camera_rig = rep.create.camera(
            position=cam_cfg['position'],
            look_at=cam_cfg['look_at']
        )
        self.render_product = rep.create.render_product(self.camera_rig, cam_cfg['resolution'])

        # 虽然这里创建了 LidarRtx，但在数据集生成中，我们主要使用 Replicator 统一获取数据
        lidar_cfg = self.cfg['sensor']['lidar']
        self.lidar = LidarRtx(
            prim_path="/World/Lidar",
            name="rtx_lidar",
            position=np.array(lidar_cfg['position']),
            orientation=np.array([1, 0, 0, 0]),
        )
        self.world.reset()

    def generate_sequences(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # ----------------------------------------
        # 1. 注册 Annotators (RGB, Shadow, PointCloud)
        # ----------------------------------------
        rgb_annotator = rep.annotators.get("rgb")
        rgb_annotator.attach(self.render_product)

        # 【新增】注册点云 Annotator
        # include_unlabelled=True 保证地面等背景也能被扫到
        pc_annotator = rep.annotators.get("pointcloud")
        pc_annotator.attach(self.render_product)

        try:
            shadow_annotator = rep.annotators.get("shadow")
            shadow_annotator.attach(self.render_product)
            use_rep_shadow = True
        except Exception:
            print("[WARN] Replicator 'shadow' annotator not found. Will use fallback.")
            use_rep_shadow = False

        object_prim_path = "/World/TargetObject"

        for seq_idx in range(self.num_seq):
            seq_name = f"seq_{seq_idx:03d}"
            seq_dir = os.path.join(self.output_dir, "sequences", seq_name)
            object_dir = os.path.join(seq_dir, "object_geometry")
            os.makedirs(object_dir, exist_ok=True)

            print(f"[{seq_idx + 1}/{self.num_seq}] Generating {seq_name}...")

            # -----------------------------
            # 2. 清理旧物体 & 刷新引擎
            # -----------------------------
            self.world.stop()
            if is_prim_path_valid(object_prim_path):
                if self.world.scene.object_exists("target_object"):
                    self.world.scene.remove_object("target_object")
                delete_prim(object_prim_path)
                omni.kit.app.get_app().update()

                # -----------------------------
            # 3. 生成新物体
            # -----------------------------
            try:
                if seq_idx % 2 == 0:
                    self.current_object = DynamicCuboid(
                        prim_path=object_prim_path,
                        name="target_object",
                        position=np.array([0, 0, 0.5]),
                        scale=np.array([0.5, 0.5, 0.5]),
                        color=np.array([1, 0, 0])
                    )
                else:
                    self.current_object = DynamicSphere(
                        prim_path=object_prim_path,
                        name="target_object",
                        position=np.array([0, 0, 0.5]),
                        radius=0.3,
                        color=np.array([0, 0, 1])
                    )

                self.world.scene.add(self.current_object)
                self.world.reset()

            except Exception as e:
                print(f"[ERROR] Object creation failed: {e}")
                omni.kit.app.get_app().update()
                continue

            # -----------------------------
            # 4. 逐帧生成
            # -----------------------------
            for frame_idx in range(self.frames_per_seq):
                frame_dir = os.path.join(seq_dir, f"frame_{frame_idx:03d}")
                os.makedirs(frame_dir, exist_ok=True)

                # 旋转物体
                angle_deg = (frame_idx / (self.frames_per_seq - 1)) * 180.0
                rot_rad = np.radians(angle_deg)
                quat = np.array([np.cos(rot_rad / 2), 0, 0, np.sin(rot_rad / 2)])
                self.current_object.set_local_pose(orientation=quat)

                # 渲染步进
                self.world.step(render=True)
                rep.orchestrator.step()

                # A. 保存 RGB
                rgb_data = rgb_annotator.get_data()
                if rgb_data is not None:
                    try:
                        Image.fromarray(rgb_data).save(os.path.join(frame_dir, "rgb.png"))
                    except:
                        pass

                # B. 保存 Shadow Mask
                if use_rep_shadow:
                    shadow_data = shadow_annotator.get_data()
                    if shadow_data is not None:
                        if shadow_data.dtype == np.float32:
                            mask_uint8 = (shadow_data * 255).astype(np.uint8)
                        else:
                            mask_uint8 = shadow_data.astype(np.uint8)
                        if len(mask_uint8.shape) == 3:
                            mask_uint8 = mask_uint8[:, :, 0]
                        Image.fromarray(mask_uint8).save(os.path.join(frame_dir, "shadow_mask.png"))

                # C. 【核心修改】保存有效点云 (Scene Pointcloud)
                pc_data = pc_annotator.get_data()
                if pc_data is not None and 'data' in pc_data:
                    # pc_data['data'] 是一个 (N, 3) 的 numpy 数组
                    points = pc_data['data']

                    if len(points) > 0:
                        # 创建 Open3D 点云对象
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)

                        # 调用你的 writer 保存为 PLY
                        save_pointcloud(pcd, os.path.join(frame_dir, "scene_pointcloud.ply"))
                    else:
                        # 如果没有点，创建一个空文件防止报错，但 CloudCompare 依然打不开
                        with open(os.path.join(frame_dir, "scene_pointcloud.ply"), "w") as f:
                            pass