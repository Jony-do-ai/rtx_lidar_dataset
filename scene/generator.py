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
from pxr import Usd, UsdGeom, Gf
# 自定义模块 (请确保 rtx_lidar_dataset/data/writer.py 存在)

def extract_synced_gt_pointcloud_from_stage(
        prim_path: str,
        out_ply_path: str,
        num_points: int = 5000,
        normalize_to_unit: bool = False,
):
    """
    从 Isaac Sim / USD Stage 中，提取指定 prim 及其所有子 Mesh 在“当前最终世界坐标系”下的点云，
    并保存为唯一 GT 文件。

    设计目标：
    1. 与阴影生成时的真实场景姿态完全一致
    2. 支持一个物体由多个 Mesh 组成
    3. 只输出一份 gt.ply，避免 dataset 误读
    4. 输出为表面均匀采样点云，而不是原始 mesh 顶点

    参数：
        prim_path:    场景中目标物体的 prim 路径，例如 self.last_rep_prim_path
        out_ply_path: 输出 ply 路径，例如 .../object_geometry/gt.ply
        num_points:   目标点数
        normalize_to_unit:
            False -> 保存世界坐标系 GT（更适合你现在的数据生成语义）
            True  -> 保存前先做中心化 + 单位球归一化
                     如果你训练时要保留 bbox_radius=1.0，建议最终用 True
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    merged_mesh = o3d.geometry.TriangleMesh()
    mesh_count = 0

    for p in Usd.PrimRange(root_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue

        usd_mesh = UsdGeom.Mesh(p)

        points_attr = usd_mesh.GetPointsAttr().Get()
        face_counts_attr = usd_mesh.GetFaceVertexCountsAttr().Get()
        face_indices_attr = usd_mesh.GetFaceVertexIndicesAttr().Get()

        if points_attr is None or face_counts_attr is None or face_indices_attr is None:
            continue

        vertices_local = np.asarray([[pt[0], pt[1], pt[2]] for pt in points_attr], dtype=np.float64)
        if vertices_local.shape[0] == 0:
            continue

        # 只处理三角面；若有非三角面则做三角扇展开
        triangles = []
        idx_cursor = 0
        for face_n in face_counts_attr:
            face_n = int(face_n)
            face = face_indices_attr[idx_cursor: idx_cursor + face_n]
            idx_cursor += face_n

            if face_n < 3:
                continue
            if face_n == 3:
                triangles.append([int(face[0]), int(face[1]), int(face[2])])
            else:
                # polygon -> triangle fan
                for k in range(1, face_n - 1):
                    triangles.append([int(face[0]), int(face[k]), int(face[k + 1])])

        if len(triangles) == 0:
            continue

        # 关键：取当前 mesh 在世界坐标系下的最终变换
        xformable = UsdGeom.Xformable(p)
        world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        vertices_world = []
        for v in vertices_local:
            vw = world_tf.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))
            vertices_world.append([vw[0], vw[1], vw[2]])
        vertices_world = np.asarray(vertices_world, dtype=np.float64)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_world)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))

        # 清理退化面，避免采样报错
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_unreferenced_vertices()

        if len(mesh_o3d.vertices) == 0 or len(mesh_o3d.triangles) == 0:
            continue

        merged_mesh += mesh_o3d
        mesh_count += 1

    if mesh_count == 0 or len(merged_mesh.vertices) == 0 or len(merged_mesh.triangles) == 0:
        raise RuntimeError(f"No valid mesh geometry found under prim: {prim_path}")

    merged_mesh.compute_vertex_normals()

    # 表面均匀采样：比直接取 mesh 顶点更适合做 GT
    pcd = merged_mesh.sample_points_uniformly(number_of_points=num_points)
    pts = np.asarray(pcd.points, dtype=np.float32)

    if normalize_to_unit:
        center = pts.mean(axis=0, keepdims=True)
        pts = pts - center
        scale = np.max(np.linalg.norm(pts, axis=1))
        scale = max(float(scale), 1e-6)
        pts = pts / scale

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    os.makedirs(os.path.dirname(out_ply_path), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd, write_ascii=True)

    print(
        f"[INFO] Saved synced GT point cloud: {out_ply_path}, "
        f"meshes={mesh_count}, points={len(np.asarray(pcd.points))}, "
        f"normalized={normalize_to_unit}"
    )

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

        # ==========================================
        # 【终极修复 1】：完全抛弃 rep.create.light
        # 使用纯原生 USD 创建平行光，彻底剥离 Replicator 的控制
        # ==========================================
        from pxr import UsdLux, UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        self.light_path = "/World/MyMovingLight"

        # 定义 USD 原生光源
        self.main_light = UsdLux.DistantLight.Define(stage, self.light_path)
        self.main_light.CreateIntensityAttr(1500)

        # 强制初始化它的旋转属性，为后面做准备
        UsdGeom.Xformable(self.main_light).AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

        # 从 self.cfg 中提取相机配置
        cam_cfg = self.cfg['sensor']['camera']
        cam_pos = cam_cfg['position']
        cam_look_at = cam_cfg['look_at']
        cam_res = cam_cfg['resolution']

        # 固定相机视角：所有帧共享同一视点，确保差分对齐
        self.camera_rig = rep.create.camera(position=cam_pos, look_at=cam_look_at)
        self.render_product = rep.create.render_product(self.camera_rig, cam_res)

        self.world.reset()

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

        semantic_annotator = rep.annotators.get("semantic_segmentation")
        semantic_annotator.attach(self.render_product)

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
                self.world.step(render=True)
                rep.orchestrator.step()

            self.world.reset()

            # ----------------------------------------
            # 在物体最终 pose 生效后，导出唯一正确的 GT 点云
            # ----------------------------------------
            gt_path = os.path.join(geom_dir, "gt.ply")

            # 先确保 pose 已真正同步到 Stage
            for _ in range(5):
                omni.kit.app.get_app().update()
                self.world.step(render=True)

            extract_synced_gt_pointcloud_from_stage(
                prim_path=self.last_rep_prim_path,
                out_ply_path=gt_path,
                num_points=5000,
                normalize_to_unit=False,  # 先保持世界坐标；训练时再统一归一化
            )

            # ==========================================
            # 2. 生成带阴影的序列帧 (永远保持阴影开启即可)
            # ==========================================
            for frame_idx in range(self.frames_per_seq):
                frame_dir = os.path.join(seq_dir, f"frame_{frame_idx:03d}")
                os.makedirs(frame_dir, exist_ok=True)

                phi = (frame_idx / self.frames_per_seq) * 360.0
                theta = 45.0

                # 直接旋转光线
                from pxr import UsdGeom
                stage = omni.usd.get_context().get_stage()
                light_prim = stage.GetPrimAtPath(self.light_path)
                if light_prim.IsValid():
                    xform = UsdGeom.Xformable(light_prim)
                    xform.ClearXformOpOrder()
                    xform.AddRotateXYZOp().Set((float(theta), 0.0, float(phi)))

                # 跑 15 帧确保光线移动到位
                for _ in range(15):
                    omni.kit.app.get_app().update()
                    self.world.step(render=True)

                # 1. 获取 RGB 图像并强制规范化到 0-255 的 uint8 类型
                raw_rgb = rgb_annotator.get_data()[..., :3].copy()
                frame_rgb = np.clip(raw_rgb if raw_rgb.max() > 1.0 else raw_rgb * 255, 0, 255).astype(np.uint8)

                # 【核心保存】：只保存训练需要的带阴影图
                Image.fromarray(frame_rgb).save(os.path.join(frame_dir, "rgb_with_shadow.png"))

                # 2. 获取并处理语义掩码 (用于区分物体和背景)
                sem_data = semantic_annotator.get_data()
                sem_img = sem_data["data"]
                target_ids = [int(sid) for sid, lbl in sem_data["info"]["idToLabels"].items() if
                              lbl.get('class') == 'target_object']

                if len(target_ids) > 0:
                    object_mask = np.isin(sem_img, target_ids)
                else:
                    object_mask = np.zeros_like(sem_img, dtype=bool)

                # 3. 计算阴影掩码 (必须保留 gray_frame 变量定义)
                # 虽然不保存 debug_gray.png，但计算 shadow_mask 需要它
                gray_frame = frame_rgb.mean(axis=2).astype(np.uint8)

                # 4. 自适应阴影阈值计算
                background_pixels = gray_frame[~object_mask]
                if len(background_pixels) > 0:
                    bg_median = np.median(background_pixels)
                    shadow_threshold = bg_median * 0.85
                else:
                    shadow_threshold = 200

                # 5. 生成并保存最终阴影掩码
                shadow_mask = np.zeros_like(gray_frame, dtype=np.uint8)
                shadow_mask[(~object_mask) & (gray_frame < shadow_threshold)] = 255

                # 【核心保存】：阴影标注
                Image.fromarray(shadow_mask).save(os.path.join(frame_dir, "shadow_mask.png"))

                # 6. 保存光线参数
                with open(os.path.join(frame_dir, "light_info.txt"), "w") as f:
                    f.write(f"theta:{theta}, phi:{phi}")