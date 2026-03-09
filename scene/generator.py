import os
import glob
import random
import numpy as np
import open3d as o3d
from PIL import Image

# Isaac Sim 核心模块
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, DynamicCuboid, DynamicSphere
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import Usd, UsdLux, Sdf
import omni.usd
from pxr import Usd, UsdLux, Sdf, UsdGeom
import omni.replicator.core as rep
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid
import omni.replicator.core as rep
# 自定义模块 (请确保 rtx_lidar_dataset/data/writer.py 存在)


class SequenceGenerator:
    def __init__(self, config, shape_library_dir):
        self.cfg = config
        self.num_seq = config['num_scenes']
        self.frames_per_seq = config['frames_per_scene']
        self.output_dir = config['output_dir']

        # 记录本地模型根目录
        self.shape_library_dir = shape_library_dir

        # 统一 XYZ 坐标系：World 坐标系，单位米
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scene()


    def setup_scene(self):
        """搭建场景：确保所有组件在同一坐标系"""
        # 地面在 Z=0
        self.ground = GroundPlane(prim_path="/World/GroundPlane", size=100.0, color=np.array([1.0, 1.0, 1.0]))

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
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DistantLight):
                # 针对不同版本的 Isaac Sim 尝试属性名
                for attr_name in ["inputs:shadow:enable", "inputs:enableShadows"]:
                    attr = prim.GetAttribute(attr_name)
                    if attr:
                        attr.Set(bool(enable))
                        # 强制更新应用程序状态，确保渲染器能够及时捕获到属性的改变
                        omni.kit.app.get_app().update()
                        return True
        print("[Warning] Could not find DistantLight to toggle shadows!")
        return False

    def generate_sequences(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 扫描 models 文件夹下的所有 usd 文件
        # 使用 glob 递归搜索所有子目录
        usd_pattern = os.path.join(self.shape_library_dir, "**/*.usd*")
        all_usd_files = glob.glob(usd_pattern, recursive=True)

        if not all_usd_files:
            print(f"[ERROR] No USD files found in {self.shape_library_dir}")
            return

        #扫描的usd文件总数
        total_models = len(all_usd_files)
        print(f"[INFO] Total models found: {total_models}. Starting batch generation...")

        # ----------------------------------------
        # 1. 注册 Annotators
        # ----------------------------------------
        rgb_annotator = rep.annotators.get("rgb")
        rgb_annotator.attach(self.render_product)

        object_prim_path = "/World/TargetObject"

        # 预热渲染器
        rep.orchestrator.step()

        for seq_idx, selected_usd in enumerate(all_usd_files):
            # 先提取 USD 所属文件夹名字
            path_parts = selected_usd.replace('\\', '/').split('/')

            if len(path_parts) >= 3 and path_parts[-2] == 'models':
                # 例如 .../03001627/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.usd
                model_name = path_parts[-3]
            elif len(path_parts) >= 2:
                model_name = path_parts[-2]
            else:
                model_name = f"model_{seq_idx}"

            # 再用这个 model_name 生成 seq_name
            seq_name = f"seq_{seq_idx:03d}_{model_name}"

            seq_dir = os.path.join(self.output_dir, "sequences", seq_name)
            geom_dir = os.path.join(seq_dir, "object_geometry")
            os.makedirs(geom_dir, exist_ok=True)

            # print(f"[{seq_idx + 1}/{self.num_seq}] Generating {seq_name}...")
            print(f"[{seq_idx + 1}/{total_models}] Processing: {selected_usd}")
            # --- 0. 清理旧物体 ---
            self.world.stop()
            from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid

            # 【完美解决累积】：精准追踪并删除上一轮由 Replicator 生成的 USD 节点
            if hasattr(self, 'last_rep_prim_path') and is_prim_path_valid(self.last_rep_prim_path):
                delete_prim(self.last_rep_prim_path)
                omni.kit.app.get_app().update()

            print(f"[{seq_idx + 1}/{total_models}] Loading model: {selected_usd}")

            # --- 1. 恢复使用 Replicator API 加载（最稳定，保证画面同步不白屏） ---
            self.current_object = rep.create.from_usd(selected_usd)

            # 强制执行一步，让 USD 节点真正在内存和场景树中生成
            rep.orchestrator.step()

            # --- 2. 找到生成的节点，自动计算包围盒并获取最大尺寸与 Z 轴偏移 ---
            from pxr import UsdGeom, Usd
            stage = omni.usd.get_context().get_stage()
            replicator_root = stage.GetPrimAtPath("/Replicator")

            rescale_factor = 1.0  # 默认缩放
            z_offset = 0.0  # 默认 Z 轴抬升量

            # 【新需求：提取所属文件夹的名字作为 model_name】
            # 将路径按 '/' 切割，通常倒数第三级是模型独立 ID（在 models 文件夹外面）
            # --- 2. 提取所属文件夹名称 ---
            path_parts = selected_usd.replace('\\', '/').split('/')
            if len(path_parts) >= 3 and path_parts[-2] == 'models':
                model_name = path_parts[-3]
            elif len(path_parts) >= 2:
                model_name = path_parts[-2]
            else:
                model_name = f"model_{seq_idx}"

            # --- 3. 【终极解决方案】用 Open3D 纯数学推导绝对准确的 Z 轴偏移与真值点云 ---
            # 这一步在后台瞬间完成，完全无视 Isaac Sim 的渲染延迟和原点偏移 bug
            original_obj_path = selected_usd.replace(".usd", ".obj")
            z_offset = 0.0

            try:
                mesh = o3d.io.read_triangle_mesh(original_obj_path)

                # 1. 计算原始网格的最大边长
                extent = mesh.get_axis_aligned_bounding_box().get_extent()
                o3d_max_dim = max(extent[0], extent[1], extent[2])

                # 2. 计算将模型缩放至 0.5 米所需的绝对比例
                o3d_scale = 0.5 / o3d_max_dim if o3d_max_dim > 0 else 1.0

                # 3. 严格模拟 Isaac Sim 的位姿变换 (绕原点转 90 度 + 缩放)
                # 注意：这里千万不要写 import numpy as np，直接用全局的 np 即可！
                R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
                mesh.rotate(R, center=(0, 0, 0))
                mesh.scale(o3d_scale, center=(0, 0, 0))

                # 4. 获取变换后，模型底部的真实 Z 坐标
                z_min = mesh.get_min_bound()[2]
                z_offset = -float(z_min)  # 这就是完美贴合地面所需的绝对抬升量！

                # 5. 执行抬升并保存完美对齐的 GT 点云
                mesh.translate((0, 0, z_offset))
                pcd = mesh.sample_points_uniformly(number_of_points=5000)

                gt_filename = f"{model_name}_gt.ply"
                o3d.io.write_point_cloud(os.path.join(geom_dir, gt_filename), pcd, write_ascii=True)

            except Exception as e:
                print(f"[ERROR] Open3D 处理与真值生成失败: {e}")

            # --- 4. 回到 Isaac Sim，计算它内部引擎需要的缩放比例 ---
            from pxr import UsdGeom, Usd
            stage = omni.usd.get_context().get_stage()
            replicator_root = stage.GetPrimAtPath("/Replicator")

            rescale_factor = 1.0
            if replicator_root:
                children = replicator_root.GetChildren()
                if children:
                    last_prim = children[-1]
                    self.last_rep_prim_path = last_prim.GetPath().pathString

                    # 获取 Isaac Sim 内部的尺寸比例 (因为它自带从 cm 到 m 的转换)
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    bbox = bbox_cache.ComputeWorldBound(last_prim)
                    range_size = bbox.GetRange().GetSize()

                    current_max_dim = max(range_size[0], range_size[1], range_size[2])
                    if current_max_dim > 0:
                        rescale_factor = 0.5 / current_max_dim

            # --- 5. 应用终极姿态：融合 Isaac内部缩放 + Open3D算出的完美抬升量 ---
            with self.current_object:
                rep.modify.pose(
                    position=(0, 0, float(z_offset)),  # 直接使用纯数学推导出的终极偏移量！
                    rotation=(90, 0, 0),
                    scale=float(rescale_factor)
                )
                rep.modify.semantics([('class', 'target_object')])

            # 强制刷新渲染器，确保动作执行完毕
            for _ in range(5):
                rep.orchestrator.step()

            self.world.reset()

            #光线问题
            self.set_light_shadows(False)
            # 3. 预热渲染器确保缓存清空
            for _ in range(10):
                self.world.step(render=True)

            # 4. 抓取图像：这张图就是“纯几何体 + 纯白背景”且无阴影
            rep.orchestrator.step()
            base_rgb = rgb_annotator.get_data()[..., :3].copy()
            base_rgb_u8 = np.clip(base_rgb, 0, 255).astype(np.uint8)
            #Image.fromarray(base_rgb_u8).save(os.path.join(geom_dir, "base_no_shadow.png"))
            Image.fromarray(base_rgb).save(os.path.join(geom_dir, "base_no_shadow.png"))

            # 5. 结束后关闭环境光，准备生成带阴影的帧
            # ambient_light.visible = False

            # --- 4. 生成唯一的点云标注 (XYZ 统一坐标) ---
            original_obj_path = selected_usd.replace(".usd", ".obj")

            try:
                mesh = o3d.io.read_triangle_mesh(original_obj_path)

                # 【对齐修复 1】：同步执行旋转 (绕 X 轴旋转 90 度)
                R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
                mesh.rotate(R, center=(0, 0, 0))

                # 【对齐修复 2】：同步执行缩放
                mesh.scale(rescale_factor, center=(0, 0, 0))

                z_min = mesh.get_min_bound()[2]
                z_offset = -float(z_min)

                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            except Exception as e:
                print(f"[ERROR] 无法从 {original_obj_path} 提取真值点云: {e}")
                pcd = o3d.geometry.PointCloud()

            # 【应用新文件名】：将原本统一的 gt_geometry.ply 换成 文件夹名字_gt.ply
            gt_filename = f"{model_name}_gt.ply"
            o3d.io.write_point_cloud(os.path.join(geom_dir, gt_filename), pcd, write_ascii=True)

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

                # 多跑几步，把缓存刷掉
                for _ in range(5):
                    self.world.step(render=True)
                    rep.orchestrator.step()

                frame_rgb = rgb_annotator.get_data()[..., :3].copy()
                Image.fromarray(frame_rgb).save(os.path.join(frame_dir, "rgb_with_shadow.png"))

                # ---------- 阴影差分 ----------
                diff = np.abs(base_rgb.astype(np.float32) - frame_rgb.astype(np.float32))
                diff_img = diff.mean(axis=2)

                # ---------- 地面 mask ----------
                ground_mask = (base_rgb.mean(axis=2) > 200)

                # ---------- 阴影 mask ----------
                shadow_mask = np.zeros_like(diff_img, dtype=np.uint8)
                shadow_mask[(ground_mask > 0) & (diff_img > 20)] = 255

                if frame_idx == 0:
                    diff0 = np.abs(base_rgb.astype(np.float32) - frame_rgb.astype(np.float32))
                    print("frame_000 mean diff =", diff0.mean(), "max diff =", diff0.max())

                Image.fromarray(shadow_mask).save(os.path.join(frame_dir, "shadow_mask.png"))

                # 保存光线参数供训练使用
                with open(os.path.join(frame_dir, "light_info.txt"), "w") as f:
                    f.write(f"theta:{theta}, phi:{phi}")