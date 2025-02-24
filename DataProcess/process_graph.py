import joblib
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from tqdm.auto import tqdm, trange
import math
import pickle as pkl
import dgl
import torch
from ruamel.yaml import YAML

import click


def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

def get_dis(point1, point2):
    dis_x = point1[0] - point2[0]
    dis_y = point1[1] - point2[1]
    dis_z = point1[2] - point2[2]
    return math.sqrt(dis_x*dis_x + dis_y*dis_y + dis_z*dis_z)

def get_amino_feature(amino):
    # all_for_assign = np.loadtxt("all_assign.txt")
    if amino == 'ALA':
        return 0
    elif amino == 'CYS':
        return 1
    elif amino == 'ASP':
        return 2
    elif amino == 'GLU':
        return 3
    elif amino == 'PHE':
        return 4
    elif amino == 'GLY':
        return 5
    elif amino == 'HIS':
        return 6
    elif amino == 'ILE':
        return 7
    elif amino == 'LYS':
        return 8
    elif amino == 'LEU':
        return 9
    elif amino == 'MET':
        return 10
    elif amino == 'ASN':
        return 11
    elif amino == 'PRO':
        return 12
    elif amino == 'GLN':
        return 13
    elif amino == 'ARG':
        return 14
    elif amino == 'SER':
        return 15
    elif amino == 'THR':
        return 16
    elif amino == 'VAL':
        return 17
    elif amino == 'TRP':
        return 18
    elif amino == 'TYR':
        return 19
    else:
        print("Amino False!")


def get_whole_pdb_graph(pdb_points, pid_list, map_pid_esm_file, residue_features, thresholds, ont, tag):
    pdb_graphs = []
    p_cnt = 0
    file_idx = 0
    for pid in tqdm(pid_list):
        p_cnt += 1
        points = pdb_points[pid]
        file_id = map_pid_esm_file[pid]
        esm_tp = read_pkl(residue_features.format(file_id))

        u_list = []
        v_list = []
        dis_list = []
        node_amino = {}
        for uid, amino_1 in enumerate(points):
            node_amino[uid] = amino_1[3]
            for vid, amino_2 in enumerate(points):
                if uid==vid:
                    continue
                dist = get_dis(amino_1, amino_2)
                if dist<=thresholds:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(dist)
        u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
        dis_list = torch.tensor(dis_list)

        graph = dgl.graph((u_list, v_list), num_nodes=len(points))
        graph.edata['dis'] = dis_list

        # graph node feature
        graph.ndata['x'] = torch.zeros(graph.num_nodes(), 1280)
        graph.ndata['aa'] = torch.zeros(graph.num_nodes(), 20)
        for node_id in range(len(points)):
            amino = points[node_id][3]
            amino_id = get_amino_feature(amino)

            graph.ndata['x'][node_id] = torch.from_numpy(esm_tp[pid][31][node_id])

            one_hot = [0.0]*20
            one_hot[amino_id] = 1.0
            graph.ndata['aa'][node_id] = torch.tensor(one_hot)

        pdb_graphs.append(graph)

        if p_cnt%5000==0:
            save_pkl('./data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)
            p_cnt = 0
            file_idx += 1
            pdb_graphs = []
    if len(pdb_graphs)>0:
        save_pkl('./data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']), default='mf')
@click.option('-t', '--thresholds', type=click.INT, default=12)

def main(data_cnf, thresholds):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf)))
    ont = data_cnf['name']

    residue_features = data_cnf['base']['residue_feature']
    map_pid_esm_file = read_pkl('./data/map_pid_esm_file.pkl')

    '''
    test
    '''
    pid_list_file = data_cnf['test']['pid_list_file']
    pdb_points_file = data_cnf['base']['pdb_points']

    with open(pid_list_file,'rb') as fr:
        used_pid_list=pkl.load(fr)

    print("Used Pid in Test: {}".format(len(used_pid_list)))

    pdb_graphs = get_whole_pdb_graph(pdb_points, used_pid_list, map_pid_esm_file, residue_features, thresholds, ont, 'test')


if __name__ == '__main__':
    main()