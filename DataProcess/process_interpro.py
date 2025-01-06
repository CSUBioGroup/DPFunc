import pandas as pd
import pickle as pkl
from pathlib import Path
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm, trange
import numpy as np

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def get_inter_whole_data(pid_list, interpro_file_path, save_file):
    rows = []
    cols = []
    data = []
    for i in trange(len(pid_list)):
        pid=pid_list[i]
        if Path.exists(Path(interpro_file_path.format(pid))):
            with open(interpro_file_path.format(pid), 'rb') as fr:
                tp = pkl.load(fr)
            vals_idx = np.argwhere(tp>0).reshape(-1)
            # print(tp.shape)
            # print(vals_idx)
            val = tp[vals_idx]
            
            rows += [i]*len(vals_idx)
            cols += vals_idx.tolist()
            data += val.tolist()
    
    # col_nodes = np.max(cols) + 1
    col_nodes = 22369
    interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(pid_list), col_nodes))
    with open(save_file, 'wb') as fw:
        pkl.dump(interpro_matrix, fw)
    
    print(interpro_matrix.shape)
    return interpro_matrix

if __name__=='__main__':
    for ont in ['mf']:
        for tag in ['test']:
            pid_list = read_pkl('./data/{}_{}_used_pid_list.pkl'.format(ont, tag))

            interpro_file_path = './data/PDB/pdb_interpro_whole_protein/{}.pkl'

            save_file = './data/{}_{}_interpro.pkl'.format(ont, tag)

            print('{} - {}'.format(ont, tag))
            get_inter_whole_data(pid_list, interpro_file_path, save_file)
