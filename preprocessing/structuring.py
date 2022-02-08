import os
import argparse
import shutil


parser = argparse.ArgumentParser(description = "Changing the file structure")
parser.add_argument('--data',       type=str, default='ds/src',             help='source Testo files')
parser.add_argument('--save_folder',type=str, default='ds/structed_box_90', help='folder for saving')
parser.add_argument('--container',  type=str, default='box',                help='container type',   choices=['box', 'pot'])
parser.add_argument('--angle',      type=str, default='90',                 help='angle of view',    choices=['90', '45'])
args = parser.parse_args()


if __name__ == '__main__':
    if os.path.exists(args.save_folder):
        raise AttributeError(f'Folder {args.save_folder} already exists!')

    os.makedirs(args.save_folder)
    days = os.listdir(args.data)
    paths_by_days = [os.path.join(args.data, day, args.container, 'IR', args.angle) for day in days]
    for day in paths_by_days:
        old_paths = [os.path.join(day, f) for f in os.listdir(day)]
        new_paths = [os.path.join(args.save_folder, f) for f in os.listdir(day)]
        [shutil.copyfile(src, dst) for src, dst in zip(old_paths, new_paths)]
    print(f'Folder {args.save_folder} A has been successfully created!')