import os
import argparse
import json
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description = "Divide dataset into train and test")
parser.add_argument('--data',       type=str, default='ds/structed_box_90_[18-23]',   help='dataset path')
parser.add_argument('--save_folder',type=str, default='ds',                           help='folder for saving json files')
args = parser.parse_args()


def getLabels(files):
    regression_labels = []
    classification_labels = []
    for file in files:
        name = file.split('.')[0]
        day = int(name.split('_')[-1:][0])
        regression_labels.append(day)
        classification_labels.append(int(day >= 5))
    return regression_labels, classification_labels


if __name__ == '__main__':
    tir_files = os.listdir(os.path.join(args.data, "TIR"))
    rgb_files = os.listdir(os.path.join(args.data, "RGB"))
    regression_labels, classification_labels = getLabels(tir_files)

    dataset = []
    for file, reg_label, clas_label in zip(tir_files, regression_labels, classification_labels):
        dataset.append({'tir': os.path.join(args.data, 'TIR', file), 
                        'rgb': os.path.join(args.data, 'RGB', file),
                        'reg_label': reg_label, 
                        'clas_label': clas_label})

    dataset_train, dataset_test = train_test_split(dataset, 
                                                   test_size=0.2, 
                                                   random_state=42, 
                                                   stratify=classification_labels, 
                                                   shuffle=True)
    
    with open(f'{args.save_folder}/train.json', 'w') as f:
        json.dump(dataset_train, f, indent=4)
    with open(f'{args.save_folder}/test.json', 'w') as f:
        json.dump(dataset_test, f, indent=4)
    with open(f'{args.save_folder}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)