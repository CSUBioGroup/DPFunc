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

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']))
@click.option('-n', '--gpu-number', type=click.INT, default=0)
@click.option('-e', '--epoch-number', type=click.INT, default=15)
@click.option('-p', '--pre-name', type=click.STRING, default='temp_model')


def main(data_cnf, gpu_number, epoch_number, pre_name):
    yaml = YAML(typ='safe')
    ont = data_cnf
    data_cnf, model_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
    device = torch.device('cuda:{}'.format(gpu_number))

    data_name, model_name = data_cnf['name'], model_cnf['name'] 
    run_name = F'{model_name}-{data_name}'
    logger.info('run_name: {}'.format(run_name))

    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Dataset: {data_name}')

    train_pid_list, train_graph, train_go = get_pdb_data(pid_list_file = data_cnf['train']['pid_list_file'],
                                                         pdb_graph_file = data_cnf['train']['pid_pdb_file'],
                                                         pid_go_file = data_cnf['train']['pid_go_file'], 
                                                         train = data_cnf['train']['train_file_count'])
    logger.info('train data done')
    valid_pid_list, valid_graph, valid_go = get_pdb_data(pid_list_file = data_cnf['valid']['pid_list_file'],
                                                         pdb_graph_file = data_cnf['valid']['pid_pdb_file'],
                                                         pid_go_file = data_cnf['valid']['pid_go_file'])
    logger.info('valid data done')
    test_pid_list, test_graph, test_go = get_pdb_data(pid_list_file = data_cnf['test']['pid_list_file'],
                                                      pdb_graph_file = data_cnf['test']['pid_pdb_file'],
                                                      pid_go_file = data_cnf['test']['pid_go_file'])
    logger.info('test data done')

    train_interpro = get_inter_whole_data(train_pid_list, data_cnf['base']['interpro_whole'], data_cnf['train']['interpro_file'])
    valid_interpro = get_inter_whole_data(valid_pid_list, data_cnf['base']['interpro_whole'], data_cnf['valid']['interpro_file'])
    test_interpro = get_inter_whole_data(test_pid_list, data_cnf['base']['interpro_whole'], data_cnf['test']['interpro_file'])

    assert len(train_pid_list)==len(train_graph)
    assert len(train_pid_list)==train_interpro.shape[0]
    assert len(train_pid_list)==len(train_go)

    assert len(valid_pid_list)==len(valid_graph)
    assert len(valid_pid_list)==valid_interpro.shape[0]
    assert len(valid_pid_list)==len(valid_go)

    assert len(test_pid_list)==len(test_graph)
    assert len(test_pid_list)==test_interpro.shape[0]
    assert len(test_pid_list)==len(test_go)

    mlb = get_mlb(Path(data_cnf['mlb']), train_go)
    labels_num = len(mlb.classes_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y = mlb.transform(train_go).astype(np.float32)
        valid_y = mlb.transform(valid_go).astype(np.float32)
        test_y  = mlb.transform(test_go).astype(np.float32)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx
        
    train_data = [(train_graph[i], i, train_y[i]) for i in range(len(train_y))]
    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=64,
        drop_last=False,
        shuffle=True)

    valid_data = [(valid_graph[i], i, valid_y[i]) for i in range(len(valid_y))]
    valid_dataloader = GraphDataLoader(
        valid_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    test_data = [(test_graph[i], i, test_y[i]) for i in range(len(test_y))]
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    del train_graph
    del test_graph
    del valid_graph
    
    logger.info('Loading Data & Model')
    
    model = combine_inter_model(inter_size=train_interpro.shape[1], 
                                inter_hid=1280, 
                                graph_size=1280, 
                                graph_hid=1280, 
                                label_num=labels_num, head=4).to(device)
    logger.info(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    loss_fn = FocalLoss()

    used_model_performance = np.array([-1.0]*3)

    for e in range(epoch_number):
        model.train()
        train_loss_vals = AverageMeter()
        for batched_graph, sample_idx, labels in tqdm(train_dataloader, leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            inter_features = (torch.from_numpy(train_interpro[sample_idx].indices).to(device).long(), 
                            torch.from_numpy(train_interpro[sample_idx].indptr).to(device).long(), 
                            torch.from_numpy(train_interpro[sample_idx].data).to(device).float())
            feats = batched_graph.ndata['x']

            logits = model(inter_features, batched_graph, feats)

            loss = loss_fn(logits, labels)
            train_loss_vals.update(loss.item(), len(labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        plus_fmax, plus_aupr, plus_t, df, valid_loss_avg = test_performance_gnn_inter(model, valid_dataloader, valid_pid_list, valid_interpro, valid_y, idx_goid, goid_idx, ont, device)
        logger.info('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(e, 
                                                                                                                                train_loss_vals.avg,
                                                                                                                                valid_loss_avg,
                                                                                                                                plus_fmax, 
                                                                                                                                plus_aupr, 
                                                                                                                                plus_t, 
                                                                                                                                df.shape))

        if e > min(used_model_performance):
                replace_ind = np.where(used_model_performance==min(used_model_performance))[0][0]
                used_model_performance[replace_ind] = e
                torch.save({'epoch': e,'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                        './save_models/{0}_{1}_{2}of{3}model.pt'.format(pre_name, ont, replace_ind, 3))
                logger.info("\t\t\t\t\tSave")

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
