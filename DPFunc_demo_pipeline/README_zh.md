# DPFunc 优化版数据处理（独立目录）

这个目录提供一个脚本化 demo，用来替代自动处理数据和生成相关文件。

## 1. 目录说明

- `templates/manifest_template.tsv`：输入数据（涵盖训练/验证/测试）
- `scripts/build_data_demo.py`：根据一个 manifest 表，自动生成 DPFunc 所需数据文件与 `configure/*.yaml`与相关特征文件。
- `scripts/run_dpfunc.py`：在指定 workspace 里运行原始 `DPFunc_main.py` / `DPFunc_pred.py`

## 2. 输入 manifest 格式

使用 TSV 或 CSV，至少要有以下列：
- `protein_id`：蛋白 ID（必须唯一）
- `split`：`train` / `valid` / `test` / `predict`（`predict` 会自动按 `test` 处理）
- `pdb_file`：PDB/PDB.GZ 文件路径（支持绝对路径、或相对仓库根目录）
- `interpro_terms`：InterPro 列表，用 `;` 分隔

可选列：
- `chain_id`：链 ID，默认 `A`
- `go_terms`：GO 列表，用 `;` 分隔（无标签可留空）

模板见：`templates/manifest_template.tsv`

## 3. 一键构建 workspace（数据处理主流程）

在项目根目录执行：

```bash
python DPFunc_demo_pipeline/scripts/build_data_demo.py \
  --manifest /path/to/your_manifest.tsv \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --pdb-dir /path/to/pdb_folder \
  --inter-idx ./data/inter_idx.pkl \
  --go-obo /path/to/go.obo
```

可选参数：
- `--mlb-source`：预测场景可传入已有 `mf_go.mlb/bp_go.mlb/cc_go.mlb`
- `--distance-threshold`：构图距离阈值，默认 `12`
- `--esm-model`：默认 `esm2_t33_650M_UR50D`
- `--repr-layer`：默认 `31`
- `--esm-batch-size`：默认 `8`
- `--device`：默认 `cuda`
- `--train-chunk-size`：默认 `5000`

## 4. 构建后会生成什么

在 `workspace` 下生成：
- `data/{ont}_train_used_pid_list.pkl` / `valid` / `test`
- `data/{ont}_train_go.txt` / `valid` / `test`
- `data/graph_features/{ont}_*_whole_pdb_part*.pkl`
- `data/interpro/{protein_id}.pkl`
- `data/{ont}_train_interpro.pkl` / `valid` / `test`
- `configure/{ont}.yaml` 与 `configure/dgg.yaml`
- `data/pdb_points.pkl`、`data/pdb_seqs.pkl`
- `data/esm_emds/esm_part_*.pkl`、`data/protein_map.pkl`


## 5. 一键训练/测试你的模型

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py train \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --epoch 15 \
  --pre-name my_model
```

结果会写到 workspace 内：
- `save_models/my_model_mf_*of3model.pt`
- `results/my_model_mf_final.pkl`

## 6. 使用已有权重做预测

1. 先把模型权重放到 workspace 的 `save_models/`，命名需匹配：
- `{pre_name}_{ont}_0of3model.pt`
- `{pre_name}_{ont}_1of3model.pt`
- `{pre_name}_{ont}_2of3model.pt`

2. 确保 workspace 下存在：
- `mlb/{ont}_go.mlb`
- `data/go.obo` (需要自己准备go.obo数据，根据实际路径替换DPFunc_main.py / DPFunc_pred.py中的路径即可)

3. 运行预测：

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py predict \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --pre-name DPFunc
```

## 7. 注意事项

- 当前 `DPFunc/data_utils.py` 中 InterPro 维度在代码里固定为 `22369`，因此默认使用 `./data/inter_idx.pkl`（长度 22369）。
- 如果你只做预测，manifest 中 `split` 用 `predict` 或 `test` 都可以，`go_terms` 可留空。
