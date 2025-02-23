import torch
import os 
import numpy as np 
import random
import glob 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu().type(torch.LongTensor), 1)
    # result = result.cuda()

    return result

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent
    
def dice_coefficient(y_pred, y_truth, eps=1e-8):
    intersection = 2 * torch.sum(torch.mul(y_pred, y_truth))
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = intersection / union 

    return dice

def dice_coefficient_airway(y_pred, y_truth, eps=1e-8):

    num_classes = y_pred.shape[1]
    y_pred = make_one_hot(torch.argmax(y_pred, 1, keepdim=True), num_classes).cuda()

    if y_truth.shape[1] != num_classes:
        y_truth = make_one_hot(y_truth, num_classes).cuda()

    dice = 0.
    for c in range(1, num_classes):
        dice += dice_coefficient(y_pred[:, c], y_truth[:, c])

    dice /= (num_classes -1 )
    
    dice2 = dice_coefficient(1 - y_pred[:, 0], 1 - y_truth[:, 0])

    return dice, dice2

def load_checkponit(net, ckpt_path, model_name='model_best.pth'):
    ckpt_list = glob.glob(os.path.join(ckpt_path, model_name))
    assert len(ckpt_list) > 0, 'No ckpt found in '+ ckpt_path 

    best_ckpt = ckpt_list[0]
    print("Best ckpt: ", best_ckpt)
    checkpoint = torch.load(best_ckpt)
    state_dict = checkpoint["state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    return net 

def plot_progress(train_loss_dict, val_loss_dict, lrs, path):
    font = {'weight': 'normal',
                    'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    x_values = list(range(len(train_loss_dict['train_loss1'])))
    colors = ['b','g','r','c','m','y','k']

    for i,(k,v) in enumerate(train_loss_dict.items()):
        ax.plot(x_values, v, color=colors[i], ls='-', label=k, linewidth=4)
        
    for i,(k,v) in enumerate(val_loss_dict.items()):
        ax.plot(x_values, v, color=colors[-(i+1)], ls='-', label=k, linewidth=4)

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()

    ax2.plot(x_values,lrs, color='g', ls='-', label="learning rate", linewidth=4)
    ax2.set_ylabel("learning rate")
    ax2.legend(loc=9)

    fig.savefig(path)
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass