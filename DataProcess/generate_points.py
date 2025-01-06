import os
import pickle as pkl
import click
from Bio import SeqIO

def get_pid_list(fasta_file):
    pid_list = []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)

    return pid_list

def read_pdb(pdb_file):
    #construct pdb ca points
    points = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('SEQRES'):
                protein_length = int(line.strip().split()[3])
            
            if line.startswith('ATOM'):
                point_type = line[12:16].strip() #col 13-16
                if point_type == 'CA':
                    x = float(line[30:38].strip()) #col 31-38
                    y = float(line[38:46].strip()) #col 39-46
                    z = float(line[46:54].strip()) #col 47-54
                    amino = line[17:20].strip() #col 18-20
                    points.append((x, y, z, amino))
    if len(points)>0:
        try:
            assert protein_length==len(points)
        except:
            return False
    return points

@click.command()
@click.option('-i', '--pid-list-file', type=click.STRING)
@click.option('-o', '--output-file', type=click.STRING)

def main(pid_list_file, output_file):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)
    
    pdb_points_info = {}
    for protein in tqdm(pid_list):
        pdb_file = './data/PDB/PDB_folder/{0}.pdb'.format(protein)
        if os.path.exists(pdb_file):
            acid_points = read_pdb(pdb_file)
            if acid_points==False:
                print("Wrong sequence length!!!")
                return False
            elif len(acid_points)==0:
                print("Empty PDB file!!!")
                return False
            else:
                pdb_points_info[protein] = acid_points
        else:
            print("Unseen proteins!!!")
            return False
    with open('./data/{}.pkl'.format(output_file), 'wb') as fw:
        pkl.dump(pdb_points_info, fw)
        print("Result save as: ./data/{}.pkl".format(output_file))


if __name__ == '__main__':
    main()