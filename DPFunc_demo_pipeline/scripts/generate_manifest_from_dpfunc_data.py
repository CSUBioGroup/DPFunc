#!/usr/bin/env python3
"""Generate manifest_template.tsv style file from DataProcess/data_dpfunc inputs."""

from __future__ import annotations

import argparse
import csv
import pickle as pkl
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SPLIT_ORDER = ["train", "valid", "test"]


def read_pid_list(file_path: Path) -> List[str]:
    pids: List[str] = []
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            pid = line.strip()
            if pid:
                pids.append(pid)
    return pids


def read_go_file(file_path: Path, ontology: str) -> Dict[str, List[str]]:
    pid_to_gos: Dict[str, List[str]] = {}
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            pid, go_id = parts[0], parts[1]
            if len(parts) >= 3 and parts[2] != ontology:
                continue
            pid_to_gos.setdefault(pid, [])
            if go_id not in pid_to_gos[pid]:
                pid_to_gos[pid].append(go_id)
    return pid_to_gos


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_interpro(file_path: Path) -> Dict[str, Sequence[str]]:
    with open(file_path, "rb") as fr:
        data = pkl.load(fr)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {file_path}, got {type(data)}")
    return data


def resolve_pdb_file(protein_id: str, af2db_dir: Path) -> Optional[Path]:
    pdb_file = af2db_dir / f"{protein_id}.pdb"
    if pdb_file.exists():
        return pdb_file
    pdb_gz_file = af2db_dir / f"{protein_id}.pdb.gz"
    if pdb_gz_file.exists():
        return pdb_gz_file
    return None


def format_pdb_path(pdb_file: Path, mode: str) -> str:
    if mode == "absolute":
        return str(pdb_file.resolve())
    if mode == "repo_relative":
        try:
            return str(pdb_file.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(pdb_file.resolve())
    return str(pdb_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest_template.tsv style file from DPFunc benchmark files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DataProcess/data_dpfunc"),
        help="Directory containing *_pid_list.txt, *_go.txt and all_protein_interpros.pkl",
    )
    parser.add_argument("--ontology", choices=["mf", "bp", "cc"], default="mf")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DPFunc_demo_pipeline/templates/mf_manifest_from_data_dpfunc.tsv"),
        help="Output manifest TSV path",
    )
    parser.add_argument(
        "--interpro-file",
        type=Path,
        default=None,
        help="Optional override for all_protein_interpros.pkl path",
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default="A",
        help="Default chain_id column value",
    )
    parser.add_argument(
        "--pdb-template",
        type=str,
        default="",
        help=(
            "Optional pdb_file template (supports {protein_id}, {split}, {ontology}); "
            "leave empty to auto-fill from AF2DB directory"
        ),
    )
    parser.add_argument(
        "--af2db-dir",
        type=Path,
        default=Path("DataProcess/AF2DB"),
        help="AF2DB directory used to auto-fill pdb_file when --pdb-template is empty",
    )
    parser.add_argument(
        "--pdb-path-mode",
        choices=["absolute", "repo_relative"],
        default="repo_relative",
        help="How to write auto-detected AF2DB pdb_file path",
    )
    parser.add_argument(
        "--allow-duplicate-across-splits",
        action="store_true",
        help="Allow the same protein_id to appear in multiple splits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    interpro_file = args.interpro_file.resolve() if args.interpro_file else data_dir / "all_protein_interpros.pkl"
    interpro_map = load_interpro(interpro_file)

    split_to_pids: Dict[str, List[str]] = {}
    split_to_go: Dict[str, Dict[str, List[str]]] = {}

    for split in SPLIT_ORDER:
        pid_file = data_dir / f"{args.ontology}_{split}_pid_list.txt"
        go_file = data_dir / f"{args.ontology}_{split}_go.txt"
        if not pid_file.exists():
            raise FileNotFoundError(f"Missing pid list: {pid_file}")
        if not go_file.exists():
            raise FileNotFoundError(f"Missing GO file: {go_file}")

        split_to_pids[split] = read_pid_list(pid_file)
        split_to_go[split] = read_go_file(go_file, args.ontology)

    if not args.allow_duplicate_across_splits:
        seen: Dict[str, str] = {}
        for split in SPLIT_ORDER:
            for pid in split_to_pids[split]:
                if pid in seen and seen[pid] != split:
                    raise ValueError(
                        f"Protein '{pid}' appears in both '{seen[pid]}' and '{split}'. "
                        "Use --allow-duplicate-across-splits to bypass."
                    )
                seen[pid] = split

    rows: List[Dict[str, str]] = []
    interpro_missing = 0
    go_missing = 0
    pdb_missing = 0

    for split in SPLIT_ORDER:
        for pid in split_to_pids[split]:
            go_terms = split_to_go[split].get(pid, [])
            if not go_terms:
                go_missing += 1

            raw_interpros = interpro_map.get(pid, [])
            if not raw_interpros:
                interpro_missing += 1
            interpros = unique_keep_order(sorted(raw_interpros))

            if args.pdb_template:
                pdb_file = args.pdb_template.format(
                    protein_id=pid,
                    split=split,
                    ontology=args.ontology,
                )
            else:
                pdb_path = resolve_pdb_file(pid, args.af2db_dir.resolve())
                if pdb_path is None:
                    pdb_missing += 1
                    pdb_file = ""
                else:
                    pdb_file = format_pdb_path(pdb_path, args.pdb_path_mode)

            rows.append(
                {
                    "protein_id": pid,
                    "split": split,
                    "pdb_file": pdb_file,
                    "chain_id": args.chain_id,
                    "go_terms": ";".join(go_terms),
                    "interpro_terms": ";".join(interpros),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(
            fw,
            fieldnames=[
                "protein_id",
                "split",
                "pdb_file",
                "chain_id",
                "go_terms",
                "interpro_terms",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {split: len(split_to_pids[split]) for split in SPLIT_ORDER}
    print("Manifest generated:")
    print(f"  output: {args.output}")
    print(f"  ontology: {args.ontology}")
    print(f"  split counts: {split_counts}")
    print(f"  rows: {len(rows)}")
    print(f"  proteins without GO terms: {go_missing}")
    print(f"  proteins without InterPro terms: {interpro_missing}")
    print(f"  proteins without PDB file path: {pdb_missing}")


if __name__ == "__main__":
    main()
