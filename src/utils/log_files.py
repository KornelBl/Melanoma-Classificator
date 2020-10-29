import os,json
import numpy as np


def save_config(config :dict):
    filename = os.path.join(get_log_dir(config) ,'config.json')
    config['output_bias'] = list(config['output_bias'])
    with open(filename, 'w' ,) as f:
        json.dump(config, f)
    config['output_bias'] = np.array(config['output_bias'])


def save_fold_iter_history(config :dict ,history :dict, fold_iter_number :int):
    filename = os.path.join(get_log_dir(config) ,str(fold_iter_number ) +'.json')
    with open(filename, 'w' ,) as f:
        json.dump(history, f)


def get_log_dir(config: dict):
    if not config["trainable_base"]:
        dir_name = "TL_"
    else:
        dir_name = ""
    dir_name = dir_name + "EffN" + str(config["effnet_version"]) + '_'
    dir_name = dir_name + str(config["image_resolution"])
    dir_name = dir_name + config['time']
    dir_path = os.path.join('logs' ,dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path