import os
import subprocess
from HRI_retarget import ROOT,SRC_ROOT,DATA_ROOT
from tqdm import tqdm
import time 
from datetime import timedelta


# 文件夹路径
folder_path = os.path.join(DATA_ROOT,"motion/human/BEAT_ZIP/beat_english_v0.2.1")
starting_time = time.time()


# 遍历文件夹及其子文件夹
for root, dirs, files in os.walk(folder_path):
    num_files = len(files)
    
    for idx, filename in tqdm(enumerate(files)):
        # 获取文件的完整路径
        file_path = os.path.join(root, filename)


        # 使用subprocess执行Shell命令
        # 示例：使用wc命令统计文件行数
        if "bvh" in file_path:
            result = subprocess.run(["python", os.path.join(SRC_ROOT,"retarget/beat_g1_inspirehands.py"), file_path], capture_output=True, text=True)
            # 打印命令输出
            print("Command output:")
            print(result.stdout)

            # 如果命令执行失败，打印错误信息
            if result.returncode != 0:
                print("Error:", result.stderr)