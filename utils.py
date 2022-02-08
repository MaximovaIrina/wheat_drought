import yaml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class ParseYAML:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [ParseYAML(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, ParseYAML(b) if isinstance(b, dict) else b)

def loadConfiguration(config):
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    return ParseYAML({**conf})

def plot_training(train_loss, test_loss, loss_name, train_acc, test_acc, metric_name, save_root):
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(loss_name)
    plt.savefig(save_root + '/loss.png', dpi=300)
    plt.close()

    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.savefig(save_root + '/metric.png', dpi=300)
    plt.close()


