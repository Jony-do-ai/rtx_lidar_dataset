print("SYNC SCRIPT STARTED")

from omni.isaac.kit import SimulationApp

# 必须先启动 Kit / Isaac 上下文
simulation_app = SimulationApp({"headless": True})
import os
import omni.client
import carb
from omni.isaac.core.utils.nucleus import get_assets_root_path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_file(src_url: str, dst_path: str):
    ensure_dir(os.path.dirname(dst_path))
    result = omni.client.copy(src_url, dst_path)
    if result != omni.client.Result.OK:
        carb.log_warn(f"[COPY FAIL] {src_url} -> {dst_path} | {result}")
    return result


def sync_folder_recursive(cloud_folder_url: str, local_folder_path: str):
    """
    递归列出 cloud_folder_url 下所有文件并逐个下载到 local_folder_path
    """
    carb.log_info(f"[LIST] {cloud_folder_url}")

    result, entries = omni.client.list(cloud_folder_url)
    if result != omni.client.Result.OK:
        carb.log_error(f"[LIST FAIL] {cloud_folder_url} | {result}")
        return

    ensure_dir(local_folder_path)

    for e in entries:
        name = e.relative_path  # 相对路径名
        if not name:
            continue

        src = cloud_folder_url.rstrip("/") + "/" + name
        dst = os.path.join(local_folder_path, name)

        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            # 子目录：递归
            sync_folder_recursive(src, dst)
        else:
            # 文件：下载
            carb.log_info(f"[DOWNLOADING] {src}")
            copy_file(src, dst)


def  list_dir(url: str):
    carb.log_info(f"[LIST] {url}")
    r, entries = omni.client.list(url)
    carb.log_info(f"[LIST RESULT] {r}")
    if r != omni.client.Result.OK:
        return []
    names = []
    for e in entries:
        # e.relative_path 是名字；flags 表示是否目录
        is_dir = bool(e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
        carb.log_info(f"  - {'[DIR] ' if is_dir else '[FILE]'}{e.relative_path}")
        names.append((e.relative_path, is_dir))
    return names

def sync_living_assets(local_base_dir=r"D:\rtx_lidar_dataset\models\isaac_models"):
    carb.log_info("===== SYNC SCRIPT STARTED =====")

    omni.client.initialize()

    cloud_root = get_assets_root_path()
    carb.log_info(f"[ASSETS ROOT] {cloud_root}")

    if not cloud_root:
        carb.log_error("无法获取 assets root。常见原因：没登录 Nucleus / Nucleus 服务不可用 / assets pack 未配置。")
        omni.client.shutdown()
        return

    # 先看看 Isaac/Props 下有什么
    # list_dir(cloud_root.rstrip("/") + "/Isaac/Props")
    # carb.log_info(f"[LOCAL BASE] {os.path.abspath(local_base_dir)}")
    # print(os.path.abspath(local_base_dir))


    # 你原来只同步 Household :contentReference[oaicite:1]{index=1}
    living_folders = [
        "/Isaac/Props/Shapes",
        "/Isaac/Props/Blocks",
        "/Isaac/Props/YCB",
        "/Isaac/Props/Mugs",
        "/Isaac/Props/Food",
    ]

    for sub_path in living_folders:
        cloud_folder = cloud_root.rstrip("/") + sub_path
        local_target = os.path.join(local_base_dir, sub_path.strip("/"))

        carb.log_info(f"\n[SYNC] {sub_path}")
        carb.log_info(f"      cloud: {cloud_folder}")
        carb.log_info(f"      local: {local_target}")

        sync_folder_recursive(cloud_folder, local_target)

    carb.log_info("===== ALL DONE =====")
    omni.client.shutdown()


if __name__ == "__main__":
    sync_living_assets(r"D:\rtx_lidar_dataset\models\isaac_models")
    simulation_app.close()
