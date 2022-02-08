import torch

class MSE:
    def __call__(self, prediction, target):
        return ((prediction - target)**2).mean()**0.5

class Fscore:
    def __call__(self, prediction, target):
        target_true = torch.sum(target).float()
        predicted_true = torch.sum(prediction).float()
        correct_true = torch.sum(prediction == prediction).float()
        recall = correct_true / target_true
        precision = correct_true / predicted_true
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
