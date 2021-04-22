import argparse
import json

import os
from bunch import Bunch
from pathlib2 import Path

_join = os.path.join

def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    change json file to dictionary
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(json_file):
    """
    解析Json文件
    solve json file
    :param json_file: 配置文件
    :return: 配置类
    """
    cfg, _ = get_config_from_json(json_file)

    cfg.finetune = eval(cfg.finetune)
    cfg.if_train = eval(cfg.if_train)

    return cfg