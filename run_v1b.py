#!/usr/bin/env python3
"""Command-line entry point for the flattened HDEG V1-B engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scientific_validation.v1.v1b_engine import V1BConfig, V1BEngine
from src.scientific_validation.v1.v1b_evidence import write_v1b_evidence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--window-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--r2-pairs", type=int, default=100_000)
    p.add_argument("--r3-max", type=int, default=50_000)
    p.add_argument("--r1-reservoir", type=int, default=100_000)
    p.add_argument("--r1-chunk", type=int, default=512)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()

    cfg = V1BConfig(
        artifact_root=a.artifact_root,
        window_root=a.window_root,
        output_root=a.output_root,
        r2_pair_count=a.r2_pairs,
        r3_max_per_condition=a.r3_max,
        r1_reservoir_size=a.r1_reservoir,
        r1_chunk_size=a.r1_chunk,
        bootstrap_replicates=a.bootstrap_replicates,
        seed=a.seed,
        bootstrap_seed=a.seed,
    )

    evidence = V1BEngine(cfg).run()
    write_v1b_evidence(a.output_root, evidence)

    print("V1-B evidence generation completed.")
    print(f"Evidence root: {a.output_root}")


if __name__ == "__main__":
    main()
