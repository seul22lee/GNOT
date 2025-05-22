#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import get_seed, get_num_params
from args import get_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch
from utils import plot_heatmap


if __name__ == "__main__":

    model_path = './data/checkpoints/heat2d_all_CGPTrel2_0520_13_22_06.pt'
    result = torch.load(model_path,map_location='cpu')


    args = result['args']

    model_dict = result['model']

    vis_component = 0 if args.component == 'all' else int(args.component)

    device = torch.device('cpu')


    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_dataset(args)

    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

    test_loader = MIODataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    loss_func = get_loss_func(args.loss_name, args, regularizer=True,  normalizer=args.normalizer)
    metric_func = get_loss_func(args.loss_name, args , regularizer=False, normalizer=args.normalizer)

    model = get_model(args,)


    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        #### test single case
        idx = 80
        g, u_p, g_u =  list(iter(test_loader))[idx]
        # u_p = u_p.unsqueeze(0)      ### test if necessary
        out = model(g, u_p, g_u)

        x, y = g.ndata['x'][:,0].cpu().numpy(), g.ndata['x'][:,1].cpu().numpy()
        pred = out[:,vis_component].squeeze().cpu().numpy()
        target =g.ndata['y'][:,vis_component].squeeze().cpu().numpy()
        err = pred - target
        print(pred)
        print(target)
        print(err)
        print(np.linalg.norm(err)/np.linalg.norm(target))





        #### choose one to visualize
        cm = plt.colormaps['rainbow']

        # 3개 subplot 나란히
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        sc0 = axes[0].scatter(x, y, c=pred, cmap=cm, s=2)
        axes[0].set_title("Prediction")
        fig.colorbar(sc0, ax=axes[0])

        sc1 = axes[1].scatter(x, y, c=target, cmap=cm, s=2)
        axes[1].set_title("Ground Truth")
        fig.colorbar(sc1, ax=axes[1])

        sc2 = axes[2].scatter(x, y, c=err, cmap=cm, s=2)
        axes[2].set_title("Error (Pred - Target)")
        fig.colorbar(sc2, ax=axes[2])

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig('pred_vs_target_vs_error.png', dpi=300)
        plt.show()



import pickle
import matplotlib.pyplot as plt

# 저장된 result.pkl 경로
pkl_path = './data/checkpoints/heat2d_all_CGPTrel2_0520_13_22_06.pkl'

# pkl 로드
with open(pkl_path, 'rb') as f:
    result = pickle.load(f)

loss_train = result['loss_train']
loss_val = result['loss_val']
lr_history = result['lr_history']

# loss plot
plt.figure()
for i in range(loss_train.shape[1]):
    plt.plot(loss_train[:, i], label=f'train loss {i}')
    plt.plot(loss_val[:, i], label=f'val loss {i}', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss History")
plt.grid(True)
plt.savefig('loss_history.png', dpi=300)
plt.show()

# lr plot
plt.figure()
plt.plot(lr_history)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('LR Schedule')
plt.grid(True)
plt.savefig('lr_schedule.png', dpi=300)
plt.show()


