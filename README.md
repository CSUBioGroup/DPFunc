# DPFunc
Accurately predicting protein function via deep learning with domain-guided structure information.



## 🚀 Latest Updates

**[March 2026] Data File Update**: The recommended workflow is now the standalone scripted pipeline under `DPFunc_demo_pipeline/`.
Legacy long-form guidance (old tutorial/training sections) has been removed from this README.

**[December 2025] Data File Update**: As the old download links currently is unavailable, we have updated the dataset used in DPFunc paper here: https://drive.google.com/file/d/1qrxbkk450GJzhVfqnAN9Ms798owrq96n/view?usp=sharing​. It consists of protein id, sequences and corresponding GO terms. You can follow our tutorial to process AF-PDB, InterPro and ESM representations, so that you can retrain our model.

**[June 2025] Data Processing Tutorial Update**: We have streamlined our data processing pipeline with a comprehensive step-by-step tutorial. Users can now easily generate all required data using our new Jupyter notebook instead of following the previous complex workflow.

---

## Dataset and Model Files

### 1. Benchmark Dataset (updated)
The dataset used in the DPFunc paper (protein IDs, sequences, GO terms):
- https://drive.google.com/file/d/1qrxbkk450GJzhVfqnAN9Ms798owrq96n/view?usp=sharing

### 2. Trained Model Weights
Pretrained DPFunc model weights:
- https://drive.google.com/file/d/1V0VTFTiB29ilbAIOZn0okBQWPlbOI3wN/view?usp=drive_link

## Updated Usage (Recommended)
Use the standalone pipeline in:
- Chinese guide: `./DPFunc_demo_pipeline/README_zh.md`
- English guide: `./DPFunc_demo_pipeline/README.md`

Typical workflow:
1. Prepare a manifest (`templates/manifest_template.tsv` format).
2. Build a standalone workspace with `build_data_demo.py`.
3. Train or predict with `run_dpfunc.py`.

Example commands:

```bash
python DPFunc_demo_pipeline/scripts/build_data_demo.py \
  --manifest /path/to/your_manifest.tsv \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --pdb-dir /path/to/pdb_folder \
  --inter-idx ./data/inter_idx.pkl \
  --go-obo /path/to/go.obo
```

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py train \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --epoch 15 \
  --pre-name my_model
```

```bash
python DPFunc_demo_pipeline/scripts/run_dpfunc.py predict \
  --workspace /path/to/demo_workspace_mf \
  --ontology mf \
  --gpu 0 \
  --pre-name DPFunc
```

## Notes
- Core model entry scripts remain unchanged: `DPFunc_main.py`, `DPFunc_pred.py`.

## Contact
- Wenkang Wang: wangwk@csu.edu.cn
- Min Li: limin@mail.csu.edu.cn

## Citation
Wang W, Shuai Y, Zeng M, et al. DPFunc: accurately predicting protein function via deep learning with domain-guided structure information. *Nature Communications*, 2025, 16(1): 70.
