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

def get_pdb_graph(pdb_points, interpro_sections, pid_list, map_pid_esm_file, residue_features, thresholds, ont, tag):
    pdb_graphs = []
    p_cnt = 0
    file_idx = 0
    for pid in tqdm(pid_list):
        p_cnt += 1
        # print(pid)
        points = pdb_points[pid]
        file_id = map_pid_esm_file[pid]
        esm_tp = read_pkl(residue_features.format(file_id))
        # print(esm_tp[pid][31].shape)

        # init section aminos
        sections = interpro_sections[pid]
        choose_aminos = set()
        vis_aminos = [False]*len(points)
        for sec in sections:
            st_amino_idx = sec[0]
            end_amino_idx = sec[1]
            for idx in range(st_amino_idx-1, end_amino_idx):
                choose_aminos.add((points[idx][0], points[idx][1], points[idx][2], points[idx][3], idx)) # (坐标xyz、氨基酸、序号)
                vis_aminos[idx] = True
        
        # find amino neighbors
        for idx, amino_1 in enumerate(points):
            if vis_aminos[idx]:
                continue
            for amino_2 in choose_aminos:
                if get_dis(amino_1, amino_2)<=thresholds:
                    choose_aminos.add((amino_1[0], amino_1[1], amino_1[2], amino_1[3], idx))
                    break
        
        # construct graph
        choose_aminos = list(choose_aminos)
        u_list = []
        v_list = []
        dis_list = []
        node_amino = {}
        for uid, amino_1 in enumerate(choose_aminos):
            node_amino[uid] = amino_1[3]
            for vid, amino_2 in enumerate(choose_aminos):
                if uid==vid:
                    continue
                dist = get_dis(amino_1, amino_2)
                if dist<=thresholds:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(dist)
        u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
        dis_list = torch.tensor(dis_list)

        graph = dgl.graph((u_list, v_list), num_nodes=len(choose_aminos))
        graph.edata['dis'] = dis_list

        # graph node feature
        graph.ndata['x'] = torch.zeros(graph.num_nodes(), 1280)
        graph.ndata['aa'] = torch.zeros(graph.num_nodes(), 20)
        for node_id in range(len(choose_aminos)):
            seq_idx = choose_aminos[node_id][4]
            amino = choose_aminos[node_id][3]
            amino_id = get_amino_feature(amino)

            graph.ndata['x'][node_id] = torch.from_numpy(esm_tp[pid][31][seq_idx])

            one_hot = [0.0]*20
            one_hot[amino_id] = 1.0
            graph.ndata['aa'][node_id] = torch.tensor(one_hot)

        # for key, val in node_amino.items():
        #     amino_id, amino_feature = get_amino_feature(val)
        #     graph.ndata['x'][key] = torch.from_numpy(amino_feature)
        #     one_hot = [0.0]*20
        #     one_hot[amino_id] = 1.0
        #     graph.ndata['aa'][key] = torch.tensor(one_hot)

        pdb_graphs.append(graph)
        # final_pid_list.append(pid)

        if p_cnt%5000==0:
            save_pkl('./data/PDB/graph_feature/{}_{}_interpro_section_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)
            p_cnt = 0
            file_idx += 1
            pdb_graphs = []
    if len(pdb_graphs)>0:
        save_pkl('./data/PDB/graph_feature/{}_{}_interpro_section_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)
    # return pdb_graphs

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
            save_pkl('/public/home/hpc224701029/walker/function/data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)
            p_cnt = 0
            file_idx += 1
            pdb_graphs = []
    if len(pdb_graphs)>0:
        save_pkl('/public/home/hpc224701029/walker/function/data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)

def get_pid_list(fasta_file):
    pid_list = []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)

    return pid_list


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
    fasta_file = data_cnf['test']['fasta_file']
    pdb_points_file = data_cnf['base']['pdb_points']

    pid_list = get_pid_list(fasta_file)

    used_pid_list = []
    with open(pdb_points_file,'rb') as fr:
        pdb_points=pkl.load(fr)
    used_pid_list = list(set(pdb_points.keys())&set(pid_list))
    print("Used Pid in Test: {}".format(len(used_pid_list)))
    save_pkl('/public/home/hpc224701029/walker/function/data/{}_{}_used_pid_list.pkl'.format(ont, 'test2'), used_pid_list)

    pdb_graphs = get_whole_pdb_graph(pdb_points, used_pid_list, map_pid_esm_file, residue_features, thresholds, ont, 'test2')
    # pdb_graphs = get_pdb_graph(pdb_points, interpro_sections, used_pid_list, map_pid_esm_file, residue_features, thresholds)

    '''
    valid
    '''
    fasta_file = data_cnf['valid']['fasta_file']
    pdb_points_file = data_cnf['base']['pdb_points']

    pid_list = get_pid_list(fasta_file)

    used_pid_list = []
    with open(pdb_points_file,'rb') as fr:
        pdb_points=pkl.load(fr)
    used_pid_list = list(set(pdb_points.keys())&set(pid_list))
    print("Used Pid in Valid: {}".format(len(used_pid_list)))
    save_pkl('/public/home/hpc224701029/walker/function/data/{}_{}_used_pid_list.pkl'.format(ont, 'test1'), used_pid_list)

    pdb_graphs = get_whole_pdb_graph(pdb_points, used_pid_list, map_pid_esm_file, residue_features, thresholds, ont, 'test1')
    # pdb_graphs = get_pdb_graph(pdb_points, interpro_sections, used_pid_list, map_pid_esm_file, residue_features, thresholds)
    
    '''
    train
    '''
    fasta_file = data_cnf['train']['fasta_file']
    pdb_points_file = data_cnf['base']['pdb_points']

    pid_list = get_pid_list(fasta_file)

    used_pid_list = []
    with open(pdb_points_file,'rb') as fr:
        pdb_points=pkl.load(fr)
    used_pid_list = list(set(pdb_points.keys())&set(pid_list))
    print("Used Pid in Train: {}".format(len(used_pid_list)))
    save_pkl('/public/home/hpc224701029/walker/function/data/{}_{}_used_pid_list.pkl'.format(ont, 'train'), used_pid_list)

    get_whole_pdb_graph(pdb_points, used_pid_list, map_pid_esm_file, residue_features, thresholds, ont, 'train')
    # get_pdb_graph(pdb_points, interpro_sections, used_pid_list, map_pid_esm_file, residue_features, thresholds, ont, 'train')


if __name__ == '__main__':
    main()