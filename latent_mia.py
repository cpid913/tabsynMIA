import torch
import numpy as np
import pandas as pd
import os
import json
import time
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import resnet_latent
import copy
import sys
import argparse

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample
from tabsyn.vae.model import Encoder_model, Model_VAE
from impute import step
from utils_train import preprocess

import src
from utils_train import make_dataset

def print_result(results):
    keys = ['auc', 'asr', 'TPR@1%FPR', 'TPR@0.1%FPR', 'threshold']
    for k, v in results.items():
        if k in keys:
            print(f'{k}: {v}')

def secmi_attack(diffusion, encoder, dataset, timestep=10, t_sec=100, batch_size=128, eval=True):
    # load splits
    train_loader = src.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    hold_out_loader = src.prepare_fast_dataloader(dataset, split='test', batch_size=batch_size)

    member_results = get_intermediate_results(diffusion, encoder, train_loader, t_sec, timestep)
    nonmember_results = get_intermediate_results(diffusion, encoder, hold_out_loader, t_sec, timestep)

    t_result = {
        'member_diffusions': member_results['internal_diffusions'],
        'member_internal_samples': member_results['internal_denoise'],
        'nonmember_diffusions': nonmember_results['internal_diffusions'],
        'nonmember_internal_samples': nonmember_results['internal_denoise'],
    }

    t_result['member_diffusions'] = t_result['member_diffusions'].to(device).float()
    t_result['member_internal_samples'] = t_result['member_internal_samples'].to(device).float()

    t_result['nonmember_diffusions'] = t_result['nonmember_diffusions'].to(device).float()
    t_result['nonmember_internal_samples'] = t_result['nonmember_internal_samples'].to(device).float()

    if eval:
        stat_result = evaluate(t_result, nns=False)
        nns_result = evaluate(t_result, nns=True)
        plt.plot([0, 1], [0, 1], linestyle='dashed', label="random guess")
        plt.plot(stat_result['fpr_list'], stat_result['tpr_list'], label=f"SecMI-stat: {stat_result['auc']:.2f}")
        plt.plot(nns_result['fpr_list'], nns_result['tpr_list'], label=f"SecMI-NNs: {nns_result['auc']:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        print('#' * 20 + ' SecMI_stat ' + '#' * 20)
        print_result(stat_result)
        print('#' * 20 + ' SecMI_NNs ' + '#' * 20)
        print_result(nns_result)

    member_t_error = ((t_result['member_internal_samples'] - t_result['member_diffusions']) ** 2).sum(dim=0)
    nonmember_t_error = ((t_result['nonmember_internal_samples'] - t_result['nonmember_diffusions']) ** 2).sum(dim=0)

    return member_t_error, nonmember_t_error


def evaluate(t_result, nns=False):
    if not nns:
        member_scores, nonmember_scores = naive_statistic_attack(t_result, metric='l2')
    else:
        member_scores, nonmember_scores = nns_attack(t_result, train_portion=0.2)
        member_scores *= -1
        nonmember_scores *= -1

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores, nonmember_scores, n_points=2000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data = {
        'member_scores': member_scores,  # for histogram
        'nonmember_scores': nonmember_scores,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPR@1%FPR': tpr_1_fpr,
        'TPR@0.1%FPR': tpr_01_fpr,
        'threshold': threshold
    }

    def print_result(results):
        keys = ['auc', 'asr', 'TPR@1%FPR', 'TPR@0.1%FPR', 'threshold']
        for k, v in results.items():
            if k in keys:
                print(f'{k}: {v}')

    return exp_data


def get_intermediate_results(diffusion, encoder, data_loader, t_sec, timestep):
    target_steps = list(range(0, t_sec, timestep))[1:]

    internal_diffusion_list = []
    internal_denoised_list = []
    for i in range(100):
        x = next(data_loader)
        x = x[0].long()
        x = x.to(device)

        num_numerical_features = dataset.X_num['train'].shape[1]
        x_num = x[:, :num_numerical_features]
        x_cat = x[:, num_numerical_features:]
        x_in = encoder(x_num, x_cat).detach().cpu().numpy()
        x_in = torch.tensor(x_in).to(device)
        x_in = x_in[:, 1:, :]
        B, num_tokens, token_dim = x_in.shape
        x_in = x_in.view(B, num_tokens * token_dim)
        x_sec = ddim_multistep(diffusion, x_in, t_c=1, target_steps=target_steps)
        x_sec = x_sec['x_t_target']
        x_sec_recon = ddim_singlestep(diffusion, x_sec, t_c=target_steps[-1],
                                      t_target=target_steps[-1] + timestep)
        x_sec_recon = ddim_singlestep(diffusion, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep,
                                      t_target=target_steps[-1])
        x_sec_recon = x_sec_recon['x_t_target']

        internal_diffusion_list.append(x_sec)
        internal_denoised_list.append(x_sec_recon)

    return {
        'internal_diffusions': torch.cat(internal_diffusion_list),
        'internal_denoise': torch.cat(internal_denoised_list)
    }


def ddim_singlestep(diffusion, x, t_c, t_target, requires_grad=False, device='cuda'):
    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = get_named_beta_schedule('linear', 1000)
    alphas = 1. - betas
    betas = torch.tensor(betas.astype('float64')).to(device)
    alphas = torch.tensor(alphas.astype('float64')).to(device)
    alphas_prod = torch.cumprod(alphas, dim=0)

    net = diffusion.denoise_fn_D
    if requires_grad:
        epsilon = net(x, t_c[0]).to(torch.float32)
    else:
        with torch.no_grad():
            epsilon = net(x, t_c[0]).to(torch.float32)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    betas_t_c = extract(betas, t=t_c, x_shape=x.shape)
    alphas_prod_t_c = extract(alphas_prod, t=t_c, x_shape=x.shape)
    alphas_prod_t_target = extract(alphas_prod, t=t_target, x_shape=x.shape)

    pred_x_0 = (x - ((1 - alphas_prod_t_c).sqrt() * epsilon)) / alphas_prod_t_c.sqrt()
    x_t_target = alphas_prod_t_target.sqrt() * pred_x_0 \
                     + (1 - alphas_prod_t_target).sqrt() * epsilon
    # pred_x_0 = (x_num - (betas_t_c/((1 - alphas_prod_t_c).sqrt()) * epsilon_num)) / (alphas_t_c.sqrt())
    # x_t_target_num = alphas_prod_t_target.sqrt() * pred_x_0 \
    #              + (1 - alphas_prod_t_target).sqrt() * epsilon_num

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }


def ddim_multistep(diffusion, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False):
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(diffusion, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def nns_attack(t_results, train_portion=0.2, device='cuda'):
    n_epoch = 15
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader, num_timestep = split_nn_datasets(t_results, train_portion=train_portion,
                                                                batch_size=batch_size)

    # initialize NNs
    model = resnet_latent.ResNet18(num_channels=1 * num_timestep * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model eval

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    model.eval()
    # generate member_scores, nonmember_scores
    member_scores = []
    nonmember_scores = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

    member_scores = torch.concat(member_scores).reshape(-1)
    nonmember_scores = torch.concat(nonmember_scores).reshape(-1)
    return member_scores, nonmember_scores


def nn_train(epoch, model, optimizer, data_loader, device='cuda'):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)

        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    # print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def split_nn_datasets(t_results, train_portion=0.1, batch_size=128):
    # split training and testing
    # [t, 25000, 3, 32, 32]
    member_diffusion = t_results['member_diffusions']
    member_sample = t_results['member_internal_samples']
    nonmember_diffusion = t_results['nonmember_diffusions']
    nonmember_sample = t_results['nonmember_internal_samples']
    # with one timestep
    # minus
    num_timestep = 1
    member_concat = (member_diffusion - member_sample).abs() ** 1
    member_concat = member_concat.reshape((member_concat.shape[0], 1, member_concat.shape[1]))
    nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** 1
    nonmember_concat = nonmember_concat.reshape((nonmember_concat.shape[0], 1, nonmember_concat.shape[1]))

    # train num
    num_train = int(member_concat.size(0) * train_portion)
    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_timestep


@torch.no_grad()
def nn_eval(model, data_loader, device='cuda'):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    # print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]


def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold


def naive_statistic_attack(t_results, metric='l2'):
    def measure(diffusion, sample, metric, device='cuda'):
        diffusion = diffusion.to(device).float()
        sample = sample.to(device).float()

        if len(diffusion.shape) == 5:
            num_timestep = diffusion.size(0)
            diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
            sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)

        if metric == 'l2':
            score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
        elif metric == 'mixed':
            score = ((diffusion - sample) ** 2).sum(dim=-1)
        else:
            raise NotImplementedError

        return score

    # member scores
    member_scores = measure(t_results['member_diffusions'], t_results['member_internal_samples'], metric=metric)
    # nonmember scores
    nonmember_scores = measure(t_results['nonmember_diffusions'], t_results['nonmember_internal_samples'],
                               metric=metric)
    return member_scores, nonmember_scores


def t_error_comparison(maxt_sec, diffusion, encoder, dataset, timestep=10):
    relative_t_column = []
    relative_t_sum = []
    for t in range(timestep * 2, maxt_sec + timestep, timestep):
        print("calculating relative t-error at timestep = " + str(t))
        member_t, nonmember_t = secmi_attack(diffusion, encoder, dataset, timestep=timestep, t_sec=t, eval=False)
        relative_t_column.append(torch.div(nonmember_t, member_t).cpu())
        relative_t_sum.append(torch.div(nonmember_t.sum(), member_t.sum()).cpu())

    # plot relative_t_sum
    plt.bar(list(map(str, range(timestep * 2, maxt_sec + timestep, timestep))), relative_t_sum, label="Hold-out Set",
            width=1, edgecolor='white')
    plt.plot([str(timestep * 2), str(maxt_sec)], [1, 1], label="Member Set", linestyle="dashed", color="black")
    plt.legend(loc="upper right")
    plt.ylabel("Relative t-error")
    plt.xlabel("Timestep")
    plt.show()


if __name__ == '__main__':
    '''
    Define paths
    '''
    # parse arguments

    dir_path = "tabsyn"
    dataset_path = "default"
    model_path = f'{dir_path}/ckpt/{dataset_path}/model.pt'
    real_data_path = f'data/{dataset_path}'
    device = 'cuda'

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] = None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = None
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    with open(f'data/{dataset_path}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    '''
    Prepare dataset
    '''

    dataset = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    d_numerical = dataset.X_num['train'].shape[1]
    categories = src.get_categories(dataset.X_cat['train'])

    '''
    Load diffusion model
    '''
    args.dataname = dataset_path
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]
    embedding_save_path = f'{dir_path}/vae/ckpt/{dataset_path}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    diffusion = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    diffusion.load_state_dict(torch.load(f'{dir_path}/ckpt/{dataset_path}/model.pt'))

    # diffusion.eval()

    '''
    Load VAE
    '''
    encoder = Encoder_model(2, d_numerical, categories, 4, n_head=1, factor=32).to(device)

    X_num, X_cat, categories, d_numerical = preprocess(real_data_path, task_type=task_type)

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    X_train_num = X_train_num.to(device)
    X_train_cat = X_train_cat.to(device)

    encoder_save_path = f'{dir_path}/vae/ckpt/{dataset_path}/encoder.pt'
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.eval()


    t_error_comparison(300, diffusion, encoder, dataset, timestep=10)
    secmi_attack(diffusion, encoder, dataset, timestep=10, t_sec=50)
