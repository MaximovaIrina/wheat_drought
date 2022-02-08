from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os

torch.manual_seed(1002)

from backbone import Backbone
from dataset import Dataset, FeachDataset
from model import MLP
from utils import *
import metrics

parser = argparse.ArgumentParser(description = "WheatDrought")
parser.add_argument('--config', default='configs/regressor.yaml')
args = parser.parse_args()
args = loadConfiguration(args.config)


def load_features(features_path, backbone, dataloader, subset):
    if os.path.exists(f'{features_path}/{subset}_feach.pt'):
        features = torch.load(f'{features_path}/{subset}_feach.pt')
        labels = torch.load(f'{features_path}/{subset}_labels.pt')
    else:
        os.makedirs(features_path, exist_ok=True)
        features, labels = [], []
        for data in dataloader:
            ndvi_t, ndvi_g, reg_label, class_label, name = data
            ndvi = ndvi_t if args.ndvi_mode == 'ndvi_t' else ndvi_g
            lbls = reg_label if args.task_mode == 'regression' else class_label
            labels.append(lbls.numpy().flatten())
            features.append(backbone(ndvi))
        features = np.vstack(features)
        labels = np.concatenate(labels)
        torch.save(features, f'{features_path}/{subset}_feach.pt')
        torch.save(labels, f'{features_path}/{subset}_labels.pt')
    features, labels = torch.Tensor(features), torch.Tensor(labels)
    return features, labels

def scale_data(train, test):
    mean, std = train.mean(dim=0), train.std(dim=0) 
    return (train-mean)/std, (test-mean)/std

if __name__ == '__main__':
    for bins in ['Static', 'LocalMins', 'MRI', 'MultiOtsu']:
        args.bins_path = f'hists/HistQuantAnalysis/HistQuant{bins}.npy'
        
        train_dataset = Dataset(args.train_list, args.transforms)
        test_dataset = Dataset(args.test_list, args.transforms)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=4)
        
        backbone = Backbone(args.ndvi_mode,
                            args.ndvi_statistic,
                            args.space_mode, 
                            args.bins_path, 
                            args.dist,
                            args.theta)

        feature_path = f'features/{args.task_mode}_{args.ndvi_mode}_{args.space_mode}_{bins}'
        train_features, train_labels = load_features(feature_path, backbone, train_dataloader, 'train')
        test_features, test_labels = load_features(feature_path, backbone, test_dataloader, 'test')

        for group in ['ALL', 'STAT', 'HIST', 'GLCM']:
            args.feature_group = group

            feature_slice = backbone.get_feature_slice(args.feature_group)
            train_feach_ds = FeachDataset(train_features, train_labels, feature_slice)
            test_feach_ds = FeachDataset(test_features, test_labels, feature_slice)
            train_dataloader = DataLoader(train_feach_ds, args.batch_size, shuffle=True, num_workers=4)
            test_dataloader = DataLoader(test_feach_ds, args.batch_size, shuffle=True, num_workers=4)
                       
            for hidden_n in np.arange(1, 36, 2):            
                args.hidden_n = hidden_n

                model = MLP(train_feach_ds.feach_len,
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
                    model.train()
                    running_loss, accuracy = 0, 0
                    for data, labels in train_dataloader:
                        optimizer.zero_grad()
                        predict = model(data)
                        loss = loss_fn(predict, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        accuracy += metric(predict, labels)
                    train_loss.append(running_loss / len(train_dataloader))
                    train_acc.append((accuracy / len(train_dataloader)).detach().numpy())

                    model.eval()
                    running_loss, accuracy = 0, 0
                    with torch.no_grad():
                        for data, labels in test_dataloader:
                            predict = model(data)
                            loss = loss_fn(predict, labels)
                            running_loss += loss.item()
                            accuracy += metric(predict, labels)
                    test_loss.append(running_loss / len(test_dataloader))
                    test_acc.append((accuracy / len(test_dataloader)).detach().numpy())

                args.exp_name = f'exps/{group}_{bins}_{hidden_n}'
                print(args.exp_name)
                os.makedirs(args.exp_name, exist_ok=True)
                common_info = {k: v for k, v in vars(backbone).items() if isinstance(v,(int, float, list, str))}
                common_info.update({'best_result': np.min(test_acc) if args.loss == 'MSELoss' else np.max(test_acc) })
                
                torch.save(common_info, args.exp_name + '/info.pt')
                torch.save(model, args.exp_name + '/model.pt')
                plot_training(train_loss, test_loss, args.loss, 
                                train_acc, test_acc, args.metric, 
                                args.exp_name)