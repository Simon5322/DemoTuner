import configparser
import shutil
import subprocess

import numpy as np
import psutil
import torch
import yaml
from gymnasium import spaces
import os

from utils.util import clear_folder


def clear_progress_result():
    progress_path = '/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/progress/'
    result_path = '/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/result/'
    clear_folder(progress_path + 's0')
    clear_folder(progress_path + 's1')

if __name__ == '__main__':
  clear_progress_result()
  dbms_name = 'mysql'
  save_folder_name = 'W20Qheavy'
  shutil.copytree(
      f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/{dbms_name}Result/{save_folder_name}/progress',
      f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/progress',
      dirs_exist_ok=True)
  # shutil.copytree(
  #     f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/{dbms_name}Result/{save_folder_name}/s1',
  #     f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/progress',
  #     dirs_exist_ok=True)
