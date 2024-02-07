import joblib
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm,trange
import math

# from nethope.psiblast_utils import blast
import pickle as pkl
import dgl
import torch

__all__ = ['get_pid_list', 'get_go_list', 'get_pid_go', 'get_pid_go_sc', 'get_data', 'output_res', 'get_mlb',
           'get_pid_go_mat', 'get_pid_go_sc_mat', 'get_ppi_idx', 'get_homo_ppi_idx', 'get_pdb_data']


def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file


def get_go_list(pid_go_file, pid_list):
    pid_go = defaultdict(list)
    with open(pid_go_file) as fp:
        for line in fp:
            # pid_go[(line_list:=line.split())[0]].append(line_list[1])
            line_list=line.split()
            pid_go[(line_list)[0]].append(line_list[1])
    return [pid_go[pid_] for pid_ in pid_list]


def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                # pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
                line_list=line.split()
                pid_go[(line_list)[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            # pid_go_sc[line_list[0]][line_list[1]] = float((line_list:=line.split('\t'))[2])
            line_list=line.split()
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list)[2])
    return dict(pid_go_sc)

def get_inter_data(pid_list_file, interpro_file_path):
    ssp_interpro = []
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    for pid in pid_list:
        with open(interpro_file_path.format(pid), 'rb') as fr:
            ssp_interpro.append(pkl.load(fr))
    return ssp_interpro

from scipy.sparse import csr_matrix
def get_inter_whole_data(pid_list, interpro_file_path, save_file):
    if Path.exists(Path(save_file)):
        with open(save_file, 'rb') as fr:
            interpro_matrix = pkl.load(fr)
        assert interpro_matrix.shape[0]==len(pid_list)
        return interpro_matrix
    
    rows = []
    cols = []
    data = []
    for i in trange(len(pid_list)):
        pid=pid_list[i]
        if Path.exists(Path(interpro_file_path.format(pid))):
            with open(interpro_file_path.format(pid), 'rb') as fr:
                tp = pkl.load(fr)
            vals_idx = np.argwhere(tp>0).reshape(-1)
            val = tp[vals_idx]
            
            rows += [i]*len(vals_idx)
            cols += vals_idx.tolist()
            data += val.tolist()
    
    # col_nodes = np.max(cols) + 1
    col_nodes = 22369
    interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(pid_list), col_nodes))
    with open(save_file, 'wb') as fw:
        pkl.dump(interpro_matrix, fw)
    
    return interpro_matrix

def get_pdb_list(pid_pdb_file, pid_list):
    with open(pid_pdb_file,'rb') as fr:
        pid_pdb=pkl.load(fr)
    return np.array([pid_pdb[pid_].numpy() for pid_ in pid_list])

def get_pdb_data(pid_list_file, pdb_graph_file, esm_feature_file, pid_go_file, train=0):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    
    if train>0:
        pdb_graphs = []
        for i in trange(train):
            with open(pdb_graph_file.format(i), 'rb') as fr:
                pdb_graphs+=pkl.load(fr)
    else:
        with open(pdb_graph_file, 'rb') as fr:
            pdb_graphs = pkl.load(fr)

    return pid_list, pdb_graphs, get_pdb_list(esm_feature_file, pid_list), get_go_list(pid_go_file, pid_list)

def padding_feature(feature_matrix, max_len):
    assert feature_matrix.shape[0]<=max_len
    return np.concatenate((feature_matrix, np.zeros((max_len-feature_matrix.shape[0], feature_matrix.shape[1]))), axis=0)


def get_pdb_featurematrix(pid_list_file, pdb_graph_file, pid_go_file, max_len=1000, train=0):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    
    esm_feature = []
    if train>0:
        for i in range(train):
            with open(pdb_graph_file.format(i), 'rb') as fr:
                pdb_graphs =pkl.load(fr)
                for t_graph in tqdm(pdb_graphs, leave=False, desc='Training File {}:'.format(i)):
                    if t_graph.ndata['x'].size(0)<max_len:
                        esm_feature.append(padding_feature(t_graph.ndata['x'].numpy(), max_len))
                    else:
                        esm_feature.append(t_graph.ndata['x'].numpy()[:max_len, :])
    else:
        with open(pdb_graph_file, 'rb') as fr:
            pdb_graphs = pkl.load(fr)
            for t_graph in tqdm(pdb_graphs, leave=False, desc='Test File:'):
                # esm_feature.append(t_graph.ndata['x'].numpy()[:max_len, :])
                if t_graph.ndata['x'].size(0)<max_len:
                    esm_feature.append(padding_feature(t_graph.ndata['x'].numpy(), max_len))
                else:
                    esm_feature.append(t_graph.ndata['x'].numpy()[:max_len, :])

    return pid_list, np.array(esm_feature), get_go_list(pid_go_file, pid_list)

def get_mean_pdb_data(pid_list_file, pdb_graph_file, pid_go_file, train=0):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    
    pdb_means = []
    if train>0:
        for i in range(train):
            with open(pdb_graph_file.format(i), 'rb') as fr:
                pdb_graphs =pkl.load(fr)
                for t_graph in tqdm(pdb_graphs, leave=False, desc='Training File {}:'.format(i)):
                    pdb_means.append(t_graph.ndata['x'].mean(axis=0).numpy())
    else:
        with open(pdb_graph_file, 'rb') as fr:
            pdb_graphs = pkl.load(fr)
            for t_graph in tqdm(pdb_graphs, leave=False, desc='Test File:'):
                pdb_means.append(t_graph.ndata['x'].mean(axis=0).numpy())

    return pid_list, np.array(pdb_means), get_go_list(pid_go_file, pid_list)

def get_base_data(pid_list_file, pid_go_file):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    
    return pid_list, get_go_list(pid_go_file, pid_list)

def get_data(fasta_file, pid_go_file, pid_pdb_file):
    pid_list = []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
    
    return pid_list, get_go_list(pid_go_file, pid_list)


def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    # if mlb_path.exists():
    #     return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=False, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat


def get_ppi_idx(pid_list, data_y, net_pid_map, data_esm):
    # print(pid_list[0])
    # num=0
    # for i,pid in enumerate(pid_list):
    #     if pid in net_pid_map:
    #         num+=1
    # print(num)
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid]) for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    
    if data_esm is None:
        esm_list=None
    else:
        esm_list=[]
        for i in pid_list_[0]:
            esm_list.append(data_esm[i])
    
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y,esm_list


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, data_esm, net_blastdb, blast_output_path):
    blast_sim = blast(net_blastdb, pid_list, fasta_file, blast_output_path)
    '''
    blast_sim: dict, blast_sim[query_pid]->{protein1: similarity1, protein2: similarity2, ...}
    '''
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    
    if data_esm is None:
        esm_list=None
    else:
        esm_list=[]
        for i in pid_list_[0]:
            esm_list.append(data_esm[i])
            
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y,esm_list
