import open3d as o3d

def save_pointcloud(pcd, filepath):
    o3d.io.write_point_cloud(filepath, pcd)
