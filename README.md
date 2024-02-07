# DeepDoguest
 Accurately predicting protein function via deep learning with domain-guided structure information

#### Prepare data
Before you train our model, you need prepare following (train/valid/test) data files: 
- protein list
- protein pre-trained embeddings
- predicted or native strucutre graph
- interpro matrix
- protein functions
- gene ontology file

The first four data files are all obtained from protein sequences, which can be generated from our DataProcess folder as following processes:
1. Assuming that you have a protein sequence file (protein.fasta) and corresponding (predicted or native) structure files (5NTC_RAT.pdb, 6PGL_SCHPO.pdb, ...):
```
>5NTC_RAT
MMTSWSDRLQNAADVPANMDKHALKKYRREAYHRVFVNRSLAMEKIKCFGFDMDYTLAVYKSPEYESLGFELTVERLVSIGYPQELLNFAYDSTFPTRGLVFDTLYGNLLKVDAYGNLLVCAHGFNFIRGP
>6PGL_SCHPO
MSVYSFSDVSLVAKALGAFVKEKSEASIKRHGVFTLAL
...
```

2. Using `generate_points.py` to generate the coordinate files of proteins `./data/pdb_points.pkl`.
```
python ./DataProcess/generate_points.py -i protein.fasta -o pdb_points
```

3. Using pre-trained language model (`esm` or other PLLMs) to generate the residue features. Since the number of proteins maybe too large, we suggest that users should partition the whole data into several parts and an additional map file `map_pid_esm_file` (`dict` format) is also needed to map the part id of each proteins. The example map file can be obtained from the `./data/PDB/map_pid_esm_file.pkl` and example data can be obtained from `./data/PDB/part_result_esm_residue_level/result_part/pdb_residue_esm_embeddings_part{part_id}.pkl`.

4. Based on `pdb_points`, `map_pid_esm_file`, and `pdb_residue_esm_embeddings_part{part_id}.pkl`, using `process_graph.py` to generate the structure graphs for training, valid, and test data. (Note: change the paths in the file)
```
python ./DataProcess/process_graph.py
```

5. Using InterProScan tool to generate the interpro items (an example can be obtained from `./data/PDB/pdb_interpro_whole_protein/{}.pkl`). Then, using `process_interpro.py` to generate the interpro sparse matrix. (Note: use your own paths in the file)
```
python ./DataProcess/process_interpro.py
```

6. Generate protein functions based on your own data. The example file can be obtained from `./data/test_cc_go.txt`.

7. Download the Gene Ontology file (`go.obo`).

8. Construct your own configure file based on previous data. (several examples can be obtained from `./configure/{mf/cc/bp}.yaml`)

#### Train our model on your own data
If you have prepared the data, you can train our model on your data as follows:
```
python DeepDoguest_main.py -d mf -n 0 -e 15 -p temp_model

arguments:
    -d: the ontology (mf/cc/bp)
    -n: gpu number (default: 0)
    -e: training epoch (default: 15)
    -p: the prefix of results (default: temp_model)
```

## Contact
Please feel free to contact us for any further questions.
- Wenkang Wang wangwk@csu.edu.cn
- Min Li limin@mail.csu.edu.cn

## References
Accurately predicting protein function via deep learning with domain-guided structure information.
