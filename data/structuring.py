import argparse
import os
import shutil


def create_struct_by_prop(src_root, dst_root):
    container, spectrum, angle = dst_root.split('/')[-1].split('_')
    days = os.listdir(src_root)
    days_paths = [os.path.join(src_root, day, container, spectrum, angle) for day in days]
    for day in days_paths:
        old_paths = [os.path.join(day, file) for file in os.listdir(day)]
        new_paths = [os.path.join(dst_root, file) for file in os.listdir(day)]
        [shutil.copyfile(src, dst) for src, dst in zip(old_paths, new_paths)]
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prop', default='box_IR_90', type=str, help='container_spectrum_angle')
    cfg = parser.parse_args()

    src_root = os.path.join(os.getcwd(), 'ds/Засуха-Пшеница-RGB-TIR')
    dst_root = os.path.join(os.getcwd(), 'ds/struct/box_IR_90')
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
        create_struct_by_prop(src_root, dst_root)
        print(f'Create {dst_root}')
    else:
        print(f'Folder {dst_root} already exists')

    ''' EXTRACT TIR & RGB FROM IRSOFT TO STRUCT_PNG FOLDER '''

