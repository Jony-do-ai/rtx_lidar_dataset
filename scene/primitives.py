import numpy as np
import open3d as o3d
import random


def sample_random_primitive():
    choice = random.choice(["sphere", "cube", "cylinder", "cone"])
    if choice == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=random.uniform(0.5, 1.5))
    elif choice == "cube":
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=random.uniform(0.5, 1.5),
            height=random.uniform(0.5, 1.5),
            depth=random.uniform(0.5, 1.5)
        )
    elif choice == "cylinder":
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=random.uniform(0.3, 1.0),
            height=random.uniform(0.5, 1.5)
        )
    elif choice == "cone":
        mesh = o3d.geometry.TriangleMesh.create_cone(
            radius=random.uniform(0.3, 1.0),
            height=random.uniform(0.5, 1.5)
        )

    # 随机旋转
    R = mesh.get_rotation_matrix_from_xyz(np.random.uniform(0, np.pi, size=3))
    mesh.rotate(R, center=mesh.get_center())

    # 随机平移
    mesh.translate(np.random.uniform(-1, 1, size=3))
    return mesh
