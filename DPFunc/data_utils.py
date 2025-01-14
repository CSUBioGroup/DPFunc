import joblib
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import trange

import pickle as pkl
from scipy.sparse import csr_matrix

__all__ = ['get_go_list', 'get_mlb', 'get_pdb_data', 'get_inter_whole_data']


def get_go_list(pid_go_file, pid_list):
    pid_go = defaultdict(list)
    with open(pid_go_file) as fp:
        for line in fp:
            line_list=line.split()
            pid_go[(line_list)[0]].append(line_list[1])
    return [pid_go[pid_] for pid_ in pid_list]

def get_pdb_data(pid_list_file, pdb_graph_file, pid_go_file, train=0):
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

    return pid_list, pdb_graphs, get_go_list(pid_go_file, pid_list)

def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(sparse_output=False, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb

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
    
    col_nodes = 22369 # this value should be the same as the length of './data/inter_idx.pkl' 
    interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(pid_list), col_nodes))
    with open(save_file, 'wb') as fw:
        pkl.dump(interpro_matrix, fw)
    
    return interpro_matrix

