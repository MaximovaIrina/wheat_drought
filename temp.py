from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch
import json
import os

from backbone import Backbone
from dataset import Dataset
from model import MLP
from utils import *
import metrics


## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====
parser = argparse.ArgumentParser(description = "WheatDrought")
parser.add_argument('--config', default='configs/regressor.yaml')
args = parser.parse_args()
args = loadConfiguration(args.config)
    
def train():
    model.train()
    running_loss = 0
    accuracy = 0

    for data in train_dataloader:
        ndvi_t, ndvi_g, reg_label, class_label, name = data
        ndvi = ndvi_t if args.ndvi_mode == 'ndvi_t' else ndvi_g
        labels = reg_label if args.task_mode == 'regression' else class_label

        optimizer.zero_grad()
        features = backbone(ndvi)
        features = torch.Tensor(features)
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        accuracy += metric(outputs, labels)

    train_loss = running_loss / len(train_dataloader)
    train_acc = (accuracy / len(train_dataloader)).detach().numpy()
    return train_loss, train_acc

def test():
    model.eval()
    running_loss = 0
    accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            ndvi_t, ndvi_g, reg_label, class_label, name = data
            ndvi = ndvi_t if args.ndvi_mode == 'ndvi_t' else ndvi_g
            labels = reg_label if args.task_mode == 'regression' else class_label

            features = backbone(ndvi)
            features = torch.Tensor(features)
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            accuracy += metric(outputs, labels)

    test_loss = running_loss / len(test_dataloader)
    test_acc = (accuracy / len(test_dataloader)).detach().numpy()
    return test_loss, test_acc

if __name__ == '__main__':
    # for group in ['ALL', 'STAT', 'HIST', 'GLCM']:
    #     for hidden_n in np.arange(1, 35, 2):
    #         for bins in ['Static', 'LocalMins', 'MRI', 'MultiOtsu']:
    for group in ['ALL',]:
        for hidden_n in np.arange(35, 36, 2):
            for bins in ['Static', ]:
                args.bins_path = f'exps/HistQuantAnalysis/HistQuant{bins}.npy'
                args.hidden_n = hidden_n
                args.feature_group = group
                args.exp_name = f'exps/{group}_{bins}_{hidden_n}'
                print(args.exp_name)
                os.makedirs(args.exp_name, exist_ok=True)

                train_dataset = Dataset(args.train_list, args.transforms)
                test_dataset = Dataset(args.test_list, args.transforms)
                train_dataloader = DataLoader(train_dataset, 3, shuffle=True, num_workers=4)
                test_dataloader = DataLoader(test_dataset, 3, shuffle=True, num_workers=4)

                backbone = Backbone(args.ndvi_mode,
                                    args.ndvi_statistic,
                                    args.space_mode, 
                                    args.bins_path, 
                                    args.dist,
                                    args.theta)

                torch.manual_seed(1002)
                model = MLP(backbone.features_len,
                            args.hidden_n,
                            args.hidden_activation,
                            args.out_n, 
                            args.out_activation)

                loss_fn = torch.nn.__getattribute__(args.loss)()
                optimizer = torch.optim.__getattribute__(args.optimizer)(model.parameters())
                metric = metrics.__getattribute__(args.metric)()

                train_loss, test_loss = [], []
                train_acc, test_acc = [], []
                for epoch in range(args.max_epoch):
                    train_log = train()
                    train_loss.append(train_log[0])
                    train_acc.append(train_log[1])

                    test_log = test()
                    test_loss.append(test_log[0])
                    test_acc.append(test_log[1])
                
                common_info = {k: v for k, v in vars(backbone).items() if isinstance(v,(int, float, list, str))}
                common_info.update({'best_result': np.min(test_acc) if args.loss == 'MSELoss' else np.max(test_acc) })
                torch.save(common_info, args.exp_name + '/info.pt')
                torch.save(model, args.exp_name + '/model.pt')
                
                plot_training(train_loss, test_loss, args.loss, 
                              train_acc, test_acc, args.metric,
                              args.exp_name)



