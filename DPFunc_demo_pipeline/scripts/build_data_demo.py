#!/usr/bin/env python3
"""Build a standalone DPFunc workspace from a simple manifest table.

This script follows DataProcess/Process_data.ipynb logic, but makes it reproducible
and easier for users to run end-to-end on their own proteins.
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle as pkl
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dgl
import esm
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


VALID_SPLITS = {"train", "valid", "test", "predict"}


def read_pkl(path: Path):
    with open(path, "rb") as fr:
        return pkl.load(fr)


def save_pkl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fw:
        pkl.dump(value, fw)


def parse_terms(raw: object) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    terms = [x.strip() for x in re.split(r"[;,|]", text) if x.strip()]
    # Keep user order while removing duplicates.
    return list(dict.fromkeys(terms))


@dataclass
class ManifestRow:
    protein_id: str
    split: str
    pdb_file: str
    chain_id: str
    go_terms: List[str]
    interpro_terms: List[str]


def load_manifest(manifest_path: Path) -> List[ManifestRow]:
    df = pd.read_csv(manifest_path, sep=None, engine="python", dtype=str).fillna("")
    required = {"protein_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    rows: List[ManifestRow] = []
    seen_pid: Dict[str, str] = {}
    for idx, row in df.iterrows():
        pid = row["protein_id"].strip()
        split = row["split"].strip().lower()
        if split == "predict":
            split = "test"
        if not pid:
            raise ValueError(f"Row {idx}: protein_id is empty")
        if split not in VALID_SPLITS:
            raise ValueError(f"Row {idx}: split '{row['split']}' is invalid")
        if pid in seen_pid:
            raise ValueError(f"Protein '{pid}' is duplicated in manifest (row {idx})")
        seen_pid[pid] = split

        rows.append(
            ManifestRow(
                protein_id=pid,
                split=split,
                pdb_file=row.get("pdb_file", "").strip(),
                chain_id=row.get("chain_id", "A").strip() or "A",
                go_terms=parse_terms(row.get("go_terms", "")),
                interpro_terms=parse_terms(row.get("interpro_terms", "")),
            )
        )

    if not rows:
        raise ValueError("Manifest has no rows")
    return rows


def resolve_pdb_path(pid: str, pdb_file: str, pdb_dir: Path, manifest_dir: Path) -> Path:
    if pdb_file:
        raw_path = Path(pdb_file)
        candidates: List[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(manifest_dir / raw_path)
            candidates.append(Path.cwd() / raw_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"PDB file for {pid} not found. Checked: {', '.join(str(x) for x in candidates)}"
        )

    candidates = [
        pdb_dir / f"{pid}.pdb",
        pdb_dir / f"{pid}.pdb.gz",
        pdb_dir / f"AF-{pid}-F1-model_v4.pdb",
        pdb_dir / f"AF-{pid}-F1-model_v4.pdb.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot resolve PDB file for {pid}. Checked: {', '.join(str(x) for x in candidates)}"
    )


def extract_sequence_and_ca_coords(pdb_path: Path, chain_id: str = "A") -> Tuple[str, List[Tuple[float, float, float]]]:
    parser = PDBParser(QUIET=True)

    if str(pdb_path).endswith(".gz"):
        with gzip.open(pdb_path, "rt") as gz_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as temp:
                temp.write(gz_file.read())
                temp_file = Path(temp.name)
        try:
            structure = parser.get_structure("protein", str(temp_file))
        finally:
            if temp_file.exists():
                temp_file.unlink()
    else:
        structure = parser.get_structure("protein", str(pdb_path))

    seq_chars: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    found_chain = False

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            found_chain = True
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                try:
                    aa = seq1(residue.resname)
                except KeyError:
                    continue
                coord = residue["CA"].get_coord()
                seq_chars.append(aa)
                coords.append((float(coord[0]), float(coord[1]), float(coord[2])))
            break

    if not found_chain:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")
    if not seq_chars or not coords:
        raise ValueError(f"No valid residues with CA atoms in {pdb_path}")
    if len(seq_chars) != len(coords):
        raise ValueError(f"Sequence/coordinate length mismatch in {pdb_path}")

    return "".join(seq_chars), coords


def maybe_compute_esm(
    proteins: Sequence[str],
    seqs: Dict[str, str],
    out_dir: Path,
    model_name: str,
    repr_layer: int,
    batch_size: int,
    device: str,
    force_rebuild: bool,
) -> Dict[str, int]:
    map_file = out_dir / "protein_map.pkl"
    if map_file.exists() and not force_rebuild:
        cached_map = read_pkl(map_file)
        if all(pid in cached_map for pid in proteins):
            return cached_map
        print("Cached ESM map is incomplete for current manifest. Rebuilding ESM embeddings...")

    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    model.eval()

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    run_device = torch.device(device if use_cuda else "cpu")
    model = model.to(run_device)

    batch_converter = alphabet.get_batch_converter()
    protein_map: Dict[str, int] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    for batch_idx in tqdm(range(0, len(proteins), batch_size), desc="ESM batches"):
        batch_proteins = proteins[batch_idx : batch_idx + batch_size]
        batch_data = [(pid, seqs[pid]) for pid in batch_proteins]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(run_device)
        with torch.no_grad():
            outputs = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_reps = outputs["representations"][repr_layer].to("cpu")

        part_dict: Dict[str, np.ndarray] = {}
        for i, pid in enumerate(batch_proteins):
            token_len = int(batch_lens[i])
            emb = token_reps[i, 1 : token_len - 1].numpy()
            if emb.shape[0] != len(seqs[pid]):
                raise ValueError(
                    f"Embedding length mismatch for {pid}: {emb.shape[0]} vs {len(seqs[pid])}"
                )
            part_dict[pid] = emb.astype(np.float32)
            protein_map[pid] = batch_idx // batch_size

        save_pkl(out_dir / f"esm_part_{batch_idx // batch_size}.pkl", part_dict)

    save_pkl(map_file, protein_map)
    return protein_map


def get_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def build_graph(coords: List[Tuple[float, float, float]], node_features: np.ndarray, threshold: float) -> dgl.DGLGraph:
    if node_features.shape[0] != len(coords):
        raise ValueError(
            f"Node feature rows ({node_features.shape[0]}) != coordinate count ({len(coords)})"
        )

    u_list: List[int] = []
    v_list: List[int] = []
    dis_list: List[float] = []

    for uid, c1 in enumerate(coords):
        for vid, c2 in enumerate(coords):
            if uid == vid:
                continue
            dist = get_distance(c1, c2)
            if dist <= threshold:
                u_list.append(uid)
                v_list.append(vid)
                dis_list.append(dist)

    u_tensor = torch.tensor(u_list, dtype=torch.int64)
    v_tensor = torch.tensor(v_list, dtype=torch.int64)
    dis_tensor = torch.tensor(dis_list, dtype=torch.float32)

    graph = dgl.graph((u_tensor, v_tensor), num_nodes=len(coords))
    graph.edata["dis"] = dis_tensor
    graph.ndata["x"] = torch.from_numpy(node_features.astype(np.float32))
    return graph


def load_esm_feature(pid: str, protein_map: Dict[str, int], esm_dir: Path, cache: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
    part_idx = protein_map[pid]
    if part_idx not in cache:
        cache[part_idx] = read_pkl(esm_dir / f"esm_part_{part_idx}.pkl")
    return cache[part_idx][pid]


def write_go_file(out_file: Path, rows: Sequence[ManifestRow], split: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fw:
        for row in rows:
            if row.split != split:
                continue
            for go in row.go_terms:
                fw.write(f"{row.protein_id} {go}\n")


def write_pid_list(out_file: Path, rows: Sequence[ManifestRow], split: str) -> List[str]:
    pids = [row.protein_id for row in rows if row.split == split]
    save_pkl(out_file, pids)
    return pids


def write_graph_files(
    out_dir: Path,
    ontology: str,
    split: str,
    pids: Sequence[str],
    coords_map: Dict[str, List[Tuple[float, float, float]]],
    protein_map: Dict[str, int],
    esm_dir: Path,
    threshold: float,
    chunk_size: int,
) -> int:
    if not pids:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cache: Dict[int, Dict[str, np.ndarray]] = {}

    if split in {"valid", "test"}:
        chunk_size = max(1, len(pids))

    chunk_id = 0
    graphs: List[dgl.DGLGraph] = []
    for idx, pid in enumerate(tqdm(pids, desc=f"Graph-{split}"), start=1):
        emb = load_esm_feature(pid, protein_map, esm_dir, cache)
        graph = build_graph(coords_map[pid], emb, threshold)
        graphs.append(graph)

        flush = idx % chunk_size == 0 or idx == len(pids)
        if flush:
            save_pkl(out_dir / f"{ontology}_{split}_whole_pdb_part{chunk_id}.pkl", graphs)
            chunk_id += 1
            graphs = []

    return chunk_id


def write_interpro_files(
    out_dir: Path,
    rows: Sequence[ManifestRow],
    pids: Sequence[str],
    inter_idx: Dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.protein_id not in pids:
            continue
        vec = np.zeros(len(inter_idx), dtype=np.float32)
        for ipr in row.interpro_terms:
            idx = inter_idx.get(ipr)
            if idx is not None:
                vec[idx] += 1.0
        save_pkl(out_dir / f"{row.protein_id}.pkl", vec)


def build_interpro_matrix(pid_list: Sequence[str], interpro_dir: Path, save_file: Path, col_nodes: int) -> None:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for ridx, pid in enumerate(pid_list):
        file_path = interpro_dir / f"{pid}.pkl"
        if not file_path.exists():
            continue
        vec = read_pkl(file_path)
        nz_idx = np.argwhere(vec > 0).reshape(-1)
        vals = vec[nz_idx]
        rows.extend([ridx] * len(nz_idx))
        cols.extend(nz_idx.tolist())
        data.extend(vals.tolist())

    mat = csr_matrix((data, (rows, cols)), shape=(len(pid_list), col_nodes))
    save_pkl(save_file, mat)


def write_config_files(
    workspace: Path,
    ontology: str,
    train_count: int,
    has_train: bool,
    has_valid: bool,
) -> None:
    cfg_dir = workspace / "configure"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    train_split = "train" if has_train else "test"
    valid_split = "valid" if has_valid else "test"

    cfg_text = f"""name: {ontology}
mlb: ./mlb/{ontology}_go.mlb
results: ./results

base:
  interpro_whole: ./data/interpro/{{}}.pkl

train:
  name: train
  pid_list_file: ./data/{ontology}_{train_split}_used_pid_list.pkl
  pid_go_file: ./data/{ontology}_{train_split}_go.txt
  pid_pdb_file: ./data/graph_features/{ontology}_{train_split}_whole_pdb_part{{}}.pkl
  train_file_count: {max(1, train_count)}
  interpro_file: ./data/{ontology}_{train_split}_interpro.pkl

valid:
  name: valid
  pid_list_file: ./data/{ontology}_{valid_split}_used_pid_list.pkl
  pid_go_file: ./data/{ontology}_{valid_split}_go.txt
  pid_pdb_file: ./data/graph_features/{ontology}_{valid_split}_whole_pdb_part0.pkl
  interpro_file: ./data/{ontology}_{valid_split}_interpro.pkl

test:
  name: test
  pid_list_file: ./data/{ontology}_test_used_pid_list.pkl
  pid_go_file: ./data/{ontology}_test_go.txt
  pid_pdb_file: ./data/graph_features/{ontology}_test_whole_pdb_part0.pkl
  interpro_file: ./data/{ontology}_test_interpro.pkl
"""

    (cfg_dir / f"{ontology}.yaml").write_text(cfg_text, encoding="utf-8")
    (cfg_dir / "dgg.yaml").write_text("name: DPFunc\n\nmodel:\n", encoding="utf-8")


def ensure_workspace_dirs(workspace: Path) -> None:
    for rel in [
        "data",
        "data/interpro",
        "data/graph_features",
        "data/esm_emds",
        "mlb",
        "configure",
        "results",
        "save_models",
    ]:
        (workspace / rel).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized DPFunc data-processing demo workspace")
    parser.add_argument("--manifest", type=Path, required=True, help="TSV/CSV with protein_id/split and optional columns")
    parser.add_argument("--workspace", type=Path, required=True, help="Output workspace directory")
    parser.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    parser.add_argument("--pdb-dir", type=Path, default=Path("."), help="Default directory for <protein_id>.pdb(.gz)")
    parser.add_argument("--inter-idx", type=Path, default=Path("data/inter_idx.pkl"), help="Path to interpro index dict")
    parser.add_argument("--go-obo", type=Path, default=None, help="Optional go.obo copy source")
    parser.add_argument("--mlb-source", type=Path, default=None, help="Optional pretrained mlb to copy into workspace/mlb")
    parser.add_argument("--distance-threshold", type=float, default=12.0)
    parser.add_argument("--esm-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-chunk-size", type=int, default=5000)
    parser.add_argument("--force-rebuild-esm", action="store_true", help="Recompute ESM embeddings even if cached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent

    rows = load_manifest(manifest_path)
    ensure_workspace_dirs(workspace)

    split_counts = {s: sum(1 for r in rows if r.split == s) for s in ["train", "valid", "test"]}
    
    all_pids = [r.protein_id for r in rows]

    '''
    1) pid list and go files
    generate {ontology}_{split}_used_pid_list.pkl and {ontology}_{split}_go.txt for split in train/valid/test
    '''
    split_to_pids: Dict[str, List[str]] = {}
    for split in ["train", "valid", "test"]:
        pids = write_pid_list(workspace / "data" / f"{args.ontology}_{split}_used_pid_list.pkl", rows, split)
        split_to_pids[split] = pids
        write_go_file(workspace / "data" / f"{args.ontology}_{split}_go.txt", rows, split)


    '''
    2) extract pdb coordinates and sequences
    generate pdb_points.pkl and pdb_seqs.pkl
    '''
    coords_map: Dict[str, List[Tuple[float, float, float]]] = {}
    seq_map: Dict[str, str] = {}
    for row in tqdm(rows, desc="PDB extraction"):
        pdb_path = resolve_pdb_path(row.protein_id, row.pdb_file, args.pdb_dir.resolve(), manifest_dir)
        seq, coords = extract_sequence_and_ca_coords(pdb_path, chain_id=row.chain_id)
        coords_map[row.protein_id] = coords
        seq_map[row.protein_id] = seq

    save_pkl(workspace / "data" / "pdb_points.pkl", coords_map)
    save_pkl(workspace / "data" / "pdb_seqs.pkl", seq_map)

    '''
    3) extract esm features
    '''
    protein_map = maybe_compute_esm(
        proteins=all_pids,
        seqs=seq_map,
        out_dir=workspace / "data" / "esm_emds",
        model_name='esm2_t33_650M_UR50D',
        repr_layer=31,
        batch_size=args.esm_batch_size,
        device=args.device,
        force_rebuild=args.force_rebuild_esm,
    )
    save_pkl(workspace / "data" / "protein_map.pkl", protein_map)

    '''
    4) graph features
    generate {ontology}_{split}_whole_pdb_part{{}}.pkl for split in train/valid/test, each containing a list of DGLGraphs
    '''
    train_count = write_graph_files(
        out_dir=workspace / "data" / "graph_features",
        ontology=args.ontology,
        split="train",
        pids=split_to_pids["train"],
        coords_map=coords_map,
        protein_map=protein_map,
        esm_dir=workspace / "data" / "esm_emds",
        threshold=args.distance_threshold,
        chunk_size=args.train_chunk_size,
    )

    for split in ["valid", "test"]:
        _ = write_graph_files(
            out_dir=workspace / "data" / "graph_features",
            ontology=args.ontology,
            split=split,
            pids=split_to_pids[split],
            coords_map=coords_map,
            protein_map=protein_map,
            esm_dir=workspace / "data" / "esm_emds",
            threshold=args.distance_threshold,
            chunk_size=max(1, len(split_to_pids[split])),
        )

    '''
    5) interpro
    '''
    inter_idx = read_pkl(args.inter_idx.resolve())
    if not isinstance(inter_idx, dict):
        raise ValueError(f"inter_idx file should be dict, got: {type(inter_idx)}")

    write_interpro_files(workspace / "data" / "interpro", rows, all_pids, inter_idx)
    for split in ["train", "valid", "test"]:
        build_interpro_matrix(
            split_to_pids[split],
            workspace / "data" / "interpro",
            workspace / "data" / f"{args.ontology}_{split}_interpro.pkl",
            col_nodes=len(inter_idx),
        )

    # 6) configs
    write_config_files(
        workspace=workspace,
        ontology=args.ontology,
        train_count=train_count,
        has_train=split_counts["train"] > 0,
        has_valid=split_counts["valid"] > 0,
    )

    # Optional dependencies for evaluation/prediction
    if args.go_obo:
        go_target = workspace / "data" / "go.obo"
        shutil.copy2(args.go_obo.resolve(), go_target)

    if args.mlb_source:
        shutil.copy2(args.mlb_source.resolve(), workspace / "mlb" / f"{args.ontology}_go.mlb")

    print("Workspace generated:")
    print(f"  {workspace}")
    print(f"  ontology={args.ontology}")
    print(f"  split counts={split_counts}")
    print(f"  train graph parts={max(1, train_count)}")
    if split_counts["train"] == 0:
        print("  NOTE: train split is empty. Use this workspace for prediction only.")
    if not (workspace / "data" / "go.obo").exists():
        print("  NOTE: data/go.obo is missing. DPFunc_main.py / DPFunc_pred.py evaluation step may fail.")


if __name__ == "__main__":
    main()
