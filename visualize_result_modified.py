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

# 비교할 모델 경로 리스트
model_names = [
    'heat2d_all_CGPTrel2_0520_13_22_06',
    'heat2d_all_MIOEGPTrel2_0526_11_23_40',
]

model_results = []

for model_name in model_names:
    print(f"\n===== Loading model: {model_name} =====")

    # Load .pt file
    model_path = f'./data/checkpoints/{model_name}.pt'
    result = torch.load(model_path, map_location='cpu')
    args = result['args']
    model_dict = result['model']
    vis_component = 0 if args.component == 'all' else int(args.component)
    device = torch.device('cpu')

    get_seed(args.seed, printout=False)
    train_dataset, test_dataset = get_dataset(args)
    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))
    test_loader = MIODataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    loss_func = get_loss_func(args.loss_name, args, regularizer=True,  normalizer=args.normalizer)
    metric_func = get_loss_func(args.loss_name, args, regularizer=False, normalizer=args.normalizer)

    model = get_model(args)
    model.load_state_dict(model_dict)
    model.eval()

    with torch.no_grad():
        idx = 80
        g, u_p, g_u = list(iter(test_loader))[idx]
        out = model(g, u_p, g_u)

        x, y = g.ndata['x'][:, 0].cpu().numpy(), g.ndata['x'][:, 1].cpu().numpy()
        pred = out[:, vis_component].squeeze().cpu().numpy()
        target = g.ndata['y'][:, vis_component].squeeze().cpu().numpy()
        err = pred - target
        rel_error = np.linalg.norm(err) / np.linalg.norm(target)

        print(f"Relative Error ({model_name}):", rel_error)

        model_results.append({
            'name': model_name,
            'x': x,
            'y': y,
            'pred': pred,
            'target': target,
            'error': err
        })

# -------------------------
# Plot Prediction Results
# -------------------------
cm = plt.colormaps['rainbow']
fig, axes = plt.subplots(len(model_results), 3, figsize=(15, 4 * len(model_results)))

for i, res in enumerate(model_results):
    sc0 = axes[i, 0].scatter(res['x'], res['y'], c=res['pred'], cmap=cm, s=2)
    axes[i, 0].set_title(f"{res['name']} - Prediction")
    fig.colorbar(sc0, ax=axes[i, 0])

    sc1 = axes[i, 1].scatter(res['x'], res['y'], c=res['target'], cmap=cm, s=2)
    axes[i, 1].set_title("Ground Truth")
    fig.colorbar(sc1, ax=axes[i, 1])

    sc2 = axes[i, 2].scatter(res['x'], res['y'], c=res['error'], cmap=cm, s=2)
    axes[i, 2].set_title("Error (Pred - Target)")
    fig.colorbar(sc2, ax=axes[i, 2])

    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('pred_vs_target_vs_error_comparison.png', dpi=300)
plt.show()

# -------------------------
# Plot Loss History
# -------------------------
plt.figure()
# 각 모델별로 개별 loss 그래프
for model_name in model_names:
    with open(f'./data/checkpoints/{model_name}.pkl', 'rb') as f:
        result = pickle.load(f)
    loss_train = result['loss_train']
    loss_val = result['loss_val']

    print(f"{model_name} - loss_train.shape: {loss_train.shape}")
    print(f"{model_name} - loss_val.shape: {loss_val.shape}")

    plt.figure()
    if loss_train.ndim == 1 and loss_val.ndim == 1:
        plt.plot(loss_train, label='train')
        plt.plot(loss_val, label='val', linestyle='--')
    elif loss_train.ndim == 2 and loss_val.ndim == 2:
        for i in range(loss_train.shape[1]):
            plt.plot(loss_train[:, i], label=f'train {i}')
            plt.plot(loss_val[:, i], label=f'val {i}', linestyle='--')
    elif loss_train.ndim == 2 and loss_val.ndim == 1:
        for i in range(loss_train.shape[1]):
            plt.plot(loss_train[:, i], label=f'train {i}')
        plt.plot(loss_val, label='val', linestyle='--')
    else:
        raise ValueError(f"Unhandled shape: train {loss_train.shape}, val {loss_val.shape}")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss History: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'loss_history_{model_name}.png', dpi=300)
    plt.show()



# -------------------------
# Plot LR Schedule
# -------------------------
plt.figure()
for model_name in model_names:
    with open(f'./data/checkpoints/{model_name}.pkl', 'rb') as f:
        result = pickle.load(f)
    plt.plot(result['lr_history'], label=f'{model_name}')

plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('LR Schedule Comparison')
plt.legend()
plt.grid(True)
plt.savefig('lr_schedule_comparison.png', dpi=300)
plt.show()
