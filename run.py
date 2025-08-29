import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from scene import Dataset
from arguments import DatasetParams, MapParams, OptimizationParams
from SLAM.utils import *
from src import config
from PMETSLAM import SLAM

torch.set_printoptions(4, sci_mode=False)
np.set_printoptions(suppress=True, precision=6)

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running PMET-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/PMETSLAM.yaml')

    if cfg['preload']:
        dataset_params = DatasetParams(parser, sentinel=True)
        dataset_params = dataset_params.extract(cfg)

        dataset = Dataset(
            dataset_params,
            shuffle=False,
            resolution_scales=dataset_params.resolution_scales,
        )
    else:
        dataset = None
    slam = SLAM(dataset, cfg, args)
    try:
        slam.run()
    except:
        slam.cleanup()



if __name__ == '__main__':
    main()
