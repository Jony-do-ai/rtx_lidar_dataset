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
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import LidarRtx

# 自定义模块 (请确保 rtx_lidar_dataset/data/writer.py 存在)
from file_io.writer import save_pointcloud


class SequenceGenerator:
    def __init__(self, config, shape_library_dir):
        self.cfg = config
        self.num_seq = config['num_scenes']
        self.frames_per_seq = config['frames_per_scene']
        self.output_dir = config['output_dir']
        self.shape_library_dir = shape_library_dir

        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()

    def setup_scene(self):
        """搭建场景"""
        self.ground = GroundPlane(prim_path="/World/GroundPlane", size=100.0, color=np.array([0.5, 0.5, 0.5]))

        # 【修改】初始化一个光源，后续通过引用来改变它
        self.main_light = rep.create.light(light_type="distant", intensity=1500)

        # 相机固定在俯视或斜视位置
        self.camera_rig = rep.create.camera(position=[0.0, -4.0, 2.5], look_at=[0.0, 0.0, 0.5])
        self.render_product = rep.create.render_product(self.camera_rig, (1024, 1024))
        cam_res = self.cfg['sensor']['camera']['resolution']
        self.render_product = rep.create.render_product(self.camera_rig, cam_res)

        # 4. Lidar (保留 Prim 结构)
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
        # 1. 注册 Annotators
        # ----------------------------------------
        rgb_annotator = rep.annotators.get("rgb")
        rgb_annotator.attach(self.render_product)

        # 【关键修复】: 显式开启 include_unlabelled=True
        # 这会强制捕获所有几何体，无论有没有 semantic label
        pc_annotator = rep.annotators.get("pointcloud", init_params={"includeUnlabelled": True})
        pc_annotator.attach(self.render_product)

        try:
            shadow_annotator = rep.annotators.get("shadow")
            shadow_annotator.attach(self.render_product)
            use_rep_shadow = True
        except:
            use_rep_shadow = False

        object_prim_path = "/World/TargetObject"

        # 预热渲染器
        rep.orchestrator.step()

        for seq_idx in range(self.num_seq):
            seq_name = f"seq_{seq_idx:03d}"
            seq_dir = os.path.join(self.output_dir, "sequences", seq_name)
            os.makedirs(os.path.join(seq_dir, "object_geometry"), exist_ok=True)

            print(f"[{seq_idx + 1}/{self.num_seq}] Generating {seq_name}...")

            # 清理旧物体
            self.world.stop()
            if is_prim_path_valid(object_prim_path):
                if self.world.scene.object_exists("target_object"):
                    self.world.scene.remove_object("target_object")
                delete_prim(object_prim_path)
                omni.kit.app.get_app().update()

                # 生成新物体
            try:
                if seq_idx % 2 == 0:
                    self.current_object = DynamicCuboid(
                        prim_path=object_prim_path, name="target_object",
                        position=np.array([0, 0, 0.5]), scale=np.array([0.5, 0.5, 0.5]), color=np.array([1, 0, 0]))
                else:
                    self.current_object = DynamicSphere(
                        prim_path=object_prim_path, name="target_object",
                        position=np.array([0, 0, 0.5]), radius=0.3, color=np.array([0, 0, 1]))

                self.world.scene.add(self.current_object)
                # 添加语义标签
                add_update_semantics(self.current_object.prim, "target_object")
                self.world.reset()
            except Exception as e:
                print(f"[ERROR] Object creation failed: {e}")
                continue

            # 预跑一步
            rep.orchestrator.step()

            for frame_idx in range(self.frames_per_seq):
                frame_dir = os.path.join(seq_dir, f"frame_{frame_idx:03d}")
                os.makedirs(frame_dir, exist_ok=True)

                # 【核心修改】：固定物体，移动光源
                # 假设每一帧我们要从不同角度照射。我们可以计算仰角(theta)和方位角(phi)
                # 例如：仰角在 30~80 度之间，方位角 0~360 度循环
                phi = (frame_idx / self.frames_per_seq) * 360.0
                theta = 45.0  # 也可以设为随机或递增

                # 更新光源旋转 (俯仰角, 偏航角, 翻滚角)
                with self.main_light:
                    rep.modify.pose(rotation=(theta, 0, phi))

                # 渲染
                self.world.step(render=False)
                rep.orchestrator.step()

                # --- 保存数据 ---

                # RGB
                rgb_data = rgb_annotator.get_data()
                if rgb_data is not None:
                    try:
                        Image.fromarray(rgb_data).save(os.path.join(frame_dir, "rgb.png"))
                    except:
                        pass

                # Shadow
                if use_rep_shadow:
                    shadow_data = shadow_annotator.get_data()
                    if shadow_data is not None:
                        if shadow_data.dtype == np.float32:
                            mask = (shadow_data * 255).astype(np.uint8)
                        else:
                            mask = shadow_data.astype(np.uint8)
                        if len(mask.shape) == 3: mask = mask[:, :, 0]
                        Image.fromarray(mask).save(os.path.join(frame_dir, "shadow_mask.png"))

                # 【点云保存核心逻辑】
                pc_data = pc_annotator.get_data()
                ply_path = os.path.join(frame_dir, "scene_pointcloud.ply")

                if pc_data is not None and 'data' in pc_data:
                    points = pc_data['data']
                    # 过滤 NaN 和 Inf (Isaac Sim 天空背景可能是 Inf)
                    valid_mask = np.isfinite(points).all(axis=1)
                    points = points[valid_mask]

                    if len(points) > 0:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        # 【重要】write_ascii=True 兼容性最好，CloudCompare 肯定能开
                        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
                    else:
                        print(f"[WARN] Frame {frame_idx}: No valid points. Writing placeholder header.")
                        # 写入一个合法的空 PLY 头，而不是空文件
                        with open(ply_path, "w") as f:
                            f.write(
                                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                else:
                    # 写入一个合法的空 PLY 头
                    with open(ply_path, "w") as f:
                        f.write(
                            "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")