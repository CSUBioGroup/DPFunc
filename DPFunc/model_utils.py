import warnings
import click
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import dgl
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger
from tqdm.auto import tqdm, trange
import networkx as nx
import torch.nn as nn
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
import pickle as pkl
import time

from DPFunc.objective import AverageMeter
from DPFunc.evaluation import new_compute_performance_deepgoplus

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss


def merge_result(cob_df_list):
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []
    
    for idx, row in cob_df_list[0].iterrows():
        save_dict['protein_id'].append(row['protein_id'])
        save_dict['gos'].append(row['gos'])
        pred_gos = {}
        # merge
        for go, score in row['predictions'].items():
            pred_gos[go] = score
        for single_df in cob_df_list[1:]:
            pred_scores = single_df[single_df['protein_id']==row['protein_id']].reset_index().loc[0, 'predictions']
            for go, score in pred_scores.items():
                pred_gos[go] += score
        # average
        avg_pred_gos = {}
        for go, score in pred_gos.items():
            avg_pred_gos[go] = score/len(cob_df_list)
        
        save_dict['predictions'].append(avg_pred_gos)
        
    df = pd.DataFrame(save_dict)
    
    return df

def test_performance_gnn_inter(model, dataloader, test_pid_list, test_interpro, test_go, idx_goid, goid_idx, ont, device, save=False, save_file=None, evaluate=True, with_relations=True):
    model.eval()
    
    true_labels = []
    pred_labels = []
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss()
    test_loss_vals = AverageMeter()
    
    for batch_idx, (x_test, sample_idx, y_test) in enumerate(dataloader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        feats = x_test.ndata['x']     
        
        inter_features = (torch.from_numpy(test_interpro[sample_idx].indices).to(device).long(), 
                          torch.from_numpy(test_interpro[sample_idx].indptr).to(device).long(), 
                          torch.from_numpy(test_interpro[sample_idx].data).to(device).float())
        
        y_pred = model(inter_features, x_test, feats)
        loss = loss_fn(y_pred, y_test)
        test_loss_vals.update(loss.item(), len(y_test))

        y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

        pred_labels.append(y_pred)
        true_labels.append(y_test)
    
    true_labels = np.vstack(true_labels)
    pred_labels = np.vstack(pred_labels)
    
    for rowid in range(pred_labels.shape[0]):
        save_dict['protein_id'].append(test_pid_list[rowid])
        
        true_gos = set()
        for goidx, goval in enumerate(test_go[rowid]):
            if goval==1:
                true_gos.add(idx_goid[goidx])
        save_dict['gos'].append(true_gos)
        
        pred_gos = {}
        for goidx, goval in enumerate(pred_labels[rowid]):
            pred_gos[idx_goid[goidx]] = goval
        save_dict['predictions'].append(pred_gos)

    df = pd.DataFrame(save_dict)
    if save:
        with open(save_file, 'wb') as fw:
            pkl.dump(df, fw)
    if evaluate:
        go_file = './data/go.obo'
        
        new_fmax, new_aupr, new_t = new_compute_performance_deepgoplus(df,go_file,ont,with_relations)

        return new_fmax, new_aupr, new_t, df, test_loss_vals.avg
    else:
        return df
