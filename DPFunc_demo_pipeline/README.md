# DPFunc Optimized Data Processing (Standalone Directory)

This directory provides a scripted demo to replace manual data preparation and related file generation.

## 1. Directory Overview

- `templates/manifest_template.tsv`: input data (covers train/valid/test)
- `scripts/build_data_demo.py`: generate DPFunc-required data files, `configure/*.yaml`, and related features from one manifest file
- `scripts/run_dpfunc.py`: run the original `DPFunc_main.py` / `DPFunc_pred.py` inside a specified workspace

## 2. Input Manifest Format

Use TSV or CSV. Required columns:
- `protein_id`: protein ID (must be unique)
- `split`: `train` / `valid` / `test` / `predict` (`predict` will be treated as `test`)
- `pdb_file`: PDB/PDB.GZ path (supports absolute path, or path relative to repo root)
- `interpro_terms`: InterPro list separated by `;`

Optional columns:
- `chain_id`: chain ID, default is `A`
- `go_terms`: GO list separated by `;` (can be empty if no labels)

Template: `templates/manifest_template.tsv`

## 3. Build Workspace in One Command (Main Data Pipeline)

Run from project root:

```bash
python DPFunc_demo_pipeline/scripts/build_data_demo.py \
  --manifest /path/to/your_manifest.tsv \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --pdb-dir /path/to/pdb_folder \
  --inter-idx ./data/inter_idx.pkl \
  --go-obo /path/to/go.obo
```

Optional arguments:
- `--mlb-source`: provide existing `mf_go.mlb/bp_go.mlb/cc_go.mlb` for prediction scenarios
- `--distance-threshold`: graph distance threshold, default `12`
- `--esm-model`: default `esm2_t33_650M_UR50D`
- `--repr-layer`: default `31`
- `--esm-batch-size`: default `8`
- `--device`: default `cuda`
- `--train-chunk-size`: default `5000`

## 4. Generated Outputs

Inside `workspace`, the pipeline generates:
- `data/{ont}_train_used_pid_list.pkl` / `valid` / `test`
- `data/{ont}_train_go.txt` / `valid` / `test`
- `data/graph_features/{ont}_*_whole_pdb_part*.pkl`
- `data/interpro/{protein_id}.pkl`
- `data/{ont}_train_interpro.pkl` / `valid` / `test`
- `configure/{ont}.yaml` and `configure/dgg.yaml`
- `data/pdb_points.pkl`, `data/pdb_seqs.pkl`
- `data/esm_emds/esm_part_*.pkl`, `data/protein_map.pkl`

## 5. Train/Test Your Model

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py train \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --epoch 15 \
  --pre-name my_model
```

Outputs are saved in the workspace:
- `save_models/my_model_mf_*of3model.pt`
- `results/my_model_mf_final.pkl`

## 6. Predict with Existing Weights

1. Put model checkpoints into workspace `save_models/` with names:
- `{pre_name}_{ont}_0of3model.pt`
- `{pre_name}_{ont}_1of3model.pt`
- `{pre_name}_{ont}_2of3model.pt`

2. Ensure the workspace has:
- `mlb/{ont}_go.mlb`
- `data/go.obo` (prepare it yourself; if needed, replace the path in `DPFunc_main.py` / `DPFunc_pred.py` accordingly)

3. Run prediction:

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py predict \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --pre-name DPFunc
```

## 7. Notes

- In `DPFunc/data_utils.py`, InterPro dimension is fixed to `22369`, so defaulting to `./data/inter_idx.pkl` (length 22369) is recommended.
- For prediction-only usage, `split` can be `predict` or `test`, and `go_terms` can be empty.
