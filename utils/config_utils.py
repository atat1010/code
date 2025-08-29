import yaml
import os

class GroupParams:
    pass


# 合并多个yaml文件，合并父子配置文件
def merge_yaml(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in b:
            if key in a:
                a[key] = merge_yaml(a[key], b[key])
            else:
                a[key] = b[key]
        return a
    else:
        return b


# 读取配置文件，并支持继承机制
def read_config(config_path):
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    while base_config["parent"] != "None" and os.path.exists(base_config["parent"]):
        with open(base_config["parent"], "r") as f:
            parent_config = yaml.safe_load(f)
        parent_config_path = parent_config["parent"]
        parent_config.update(base_config)
        base_config = parent_config
        base_config["parent"] = parent_config_path
    group = GroupParams()
    for k, v in base_config.items():
        setattr(group, k.lstrip("_"), v)
    return group
