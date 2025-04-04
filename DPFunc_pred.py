from ruamel.yaml import YAML
from logzero import logger
from pathlib import Path
import warnings

import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

from DPFunc.data_utils import get_pdb_data, get_mlb, get_inter_whole_data
from DPFunc.models import combine_inter_model
from DPFunc.objective import AverageMeter
from DPFunc.model_utils import test_performance_gnn_inter, merge_result, FocalLoss
from DPFunc.evaluation import new_compute_performance_deepgoplus

import os
import pickle as pkl
import click
from tqdm.auto import tqdm
import joblib

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']))
@click.option('-n', '--gpu-number', type=click.INT, default=0)
@click.option('-p', '--pre-name', type=click.STRING, default='temp_model')


def main(data_cnf, gpu_number, pre_name):
    yaml = YAML(typ='safe')
    ont = data_cnf
    data_cnf, model_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
    device = torch.device('cuda:{}'.format(gpu_number))

    data_name, model_name = data_cnf['name'], model_cnf['name']

    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Dataset: {data_name}')

    test_pid_list, test_graph, test_go = get_pdb_data(pid_list_file = data_cnf['test']['pid_list_file'],
                                                      pdb_graph_file = data_cnf['test']['pid_pdb_file'],
                                                      pid_go_file = data_cnf['test']['pid_go_file'])
    logger.info('test data done')

    test_interpro = get_inter_whole_data(test_pid_list, data_cnf['base']['interpro_whole'], data_cnf['test']['interpro_file'])

    assert len(test_pid_list)==len(test_graph)
    assert len(test_pid_list)==test_interpro.shape[0]
    assert len(test_pid_list)==len(test_go)

    mlb = joblib.load(Path(data_cnf['mlb']))
    labels_num = len(mlb.classes_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        test_y  = mlb.transform(test_go).astype(np.float32)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx

    test_data = [(test_graph[i], i, test_y[i]) for i in range(len(test_y))]
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)
    
    logger.info('Loading Data & Model')
    
    model = combine_inter_model(inter_size=test_interpro.shape[1], 
                                inter_hid=1280, 
                                graph_size=1280, 
                                graph_hid=1280, 
                                label_num=labels_num, head=4).to(device)
    logger.info(model)
    
    cob_pred_df = []
    for i_t_min in range(3):
        if os.path.exists('./save_models/{0}_{1}_{2}of{3}model.pt'.format(pre_name, ont, i_t_min, 3)):
            checkpoint = torch.load('./save_models/{0}_{1}_{2}of{3}model.pt'.format(pre_name, ont, i_t_min, 3))
            model.load_state_dict(checkpoint['model_state_dict'])
            pred_df = test_performance_gnn_inter(model, test_dataloader, test_pid_list, test_interpro, test_y, idx_goid, goid_idx, ont, device, 
                                        save=True, save_file='./results/{0}_{1}_{2}of{3}model.pkl'.format(pre_name, ont, i_t_min, 3), evaluate=False)
            cob_pred_df.append(pred_df)
            print(i_t_min, 'epoch:', checkpoint['epoch'], pred_df.shape)

    final_result = merge_result(cob_pred_df)
    with open('./results/{}_{}_final.pkl'.format(pre_name, ont), 'wb') as fw:
        pkl.dump(final_result, fw)
    logger.info("Done")
    go_file = './data/go.obo'
    new_fmax, new_aupr, new_t = new_compute_performance_deepgoplus(final_result, go_file, ont)
    logger.info('Final Result: plus_Fmax on test: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}'.format(new_fmax, new_aupr, new_t))

if __name__ == '__main__':
    main()
