import open3d as o3d

def mesh_to_pointcloud(geom, num_points=100000):
    """
    Accepts TriangleMesh or PointCloud
    Returns PointCloud
    """

    # Case 1: Triangle mesh with faces
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if len(geom.triangles) == 0:
            raise ValueError("TriangleMesh has no triangles")
        return geom.sample_points_uniformly(number_of_points=num_points)

    # Case 2: Already point cloud
    if isinstance(geom, o3d.geometry.PointCloud):
        return geom

    raise TypeError("Unsupported geometry type")
