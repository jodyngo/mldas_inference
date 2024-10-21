# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

from mmengine.logging import print_log
from mmdet3d.apis import LidarSeg3DInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd_dir', help='Directory containing point cloud files')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args


def main():
    # Parse arguments and set up inference configurations
    init_args, call_args = parse_args()

    # Initialize the inferencer
    inferencer = LidarSeg3DInferencer(**init_args)

    # Get the directory of point cloud files
    pcd_dir = call_args.pop('pcd_dir')

    # Loop over all .bin files in the specified directory
    for pcd_file in os.listdir(pcd_dir):
        if pcd_file.endswith('.bin'):
            # Set the current point cloud file path
            call_args['inputs'] = dict(points=os.path.join(pcd_dir, pcd_file))
            
            # Run inference for the current file
            inferencer(**call_args)

            # Log the output location
            if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                                   and call_args['no_save_pred']):
                print_log(
                    f'Results for {pcd_file} have been saved at {call_args["out_dir"]}',
                    logger='current')


if __name__ == '__main__':
    main()
