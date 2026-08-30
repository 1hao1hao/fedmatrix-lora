# 文件: convert_dataset.py
import os
from datasets import load_dataset

# 1. 再次指定之前的缓存路径 (确保能利用你之前传上去的 35G 缓存)
project_root = os.getcwd()
cache_dir = os.path.join(project_root, "hf_cache")

print(f"Loading from cache: {cache_dir}")

# 2. 加载数据集 (此时登录节点有网，或者利用缓存)
# 如果缓存有效，这里很快；如果无效，它会下载
dataset = load_dataset("cifar100", cache_dir=cache_dir)

# 3. [核心步骤] 保存为本地磁盘格式
# 这会把数据解压并保存为 Arrow 格式到 'cifar100_local' 文件夹
# 以后读取它就像读取普通文件一样，完全不需要网络
save_path = os.path.join(project_root, "cifar100_local")
print(f"Saving to local disk: {save_path} ...")
dataset.save_to_disk(save_path)

print("Conversion complete!")