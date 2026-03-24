Isaac sim 的python环境安装库："D:\isaac-sim\python.bat" -m pip install open3d numpy

关于shapenet数据集下载
1、为下载专门建了一个新的anaconda环境；在该环境中安装hugging face下载插件：
pip install huggingface_hub
huggingface-cli login
2、在该环境下运行我写的下载数据集脚本
python D:\rtx_lidar_dataset\shapeNet_download\shapeNet_download.py

数据集解压
python unzip_shapenet.py

数据集obj转usd
需要用到isaac Sim组件，所以要用Isaac的python
直接系统原生cmd，执行：
"D:\isaac-sim\python.bat" D:\rtx_lidar_dataset\shapeNet_download\convert_to_usd.py
我已经写了一个可断点的循环执行脚本auto_run.bat，每次转2000个，直接双击执行就行了。
