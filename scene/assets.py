import os
import open3d as o3d
import random

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