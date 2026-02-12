import os
import open3d as o3d
import random
import omni.client

def load_random_library_model(library_dir):

    files = [f for f in os.listdir(library_dir) if f.endswith((".obj",".ply", ".usd"))]
    if not files:
        raise ValueError(f"No 3D models in {library_dir}")
    model_file = random.choice(files)
    mesh = o3d.io.read_triangle_mesh(os.path.join(library_dir, model_file))
    return mesh

def load_geometry(path):
    geom = o3d.io.read_triangle_mesh(path)
    if len(geom.triangles) > 0:
        geom.compute_vertex_normals()
        return geom

    # fallback: try point cloud
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) > 0:
        return pcd

    raise RuntimeError(f"Invalid geometry: {path}")


def get_usd_assets_from_nucleus(nucleus_path):
    """从 Nucleus 云端目录获取所有 .usd 文件列表"""
    result, entries = omni.client.list(nucleus_path)
    if result != omni.client.Result.OK:
        return []

    # 过滤出 usd 文件
    usd_files = [os.path.join(nucleus_path, e.relative_path)
                 for e in entries if e.relative_path.endswith((".usd", ".usda", ".usdc"))]
    return usd_files