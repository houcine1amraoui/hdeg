"""V1-B Scientific Validation Engine.

This module computes V1 representation-quality evidence only.
It deliberately does not synthesize scientific claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np

from v1b_io import (
    discover_shards,
    load_representation_shard,
    load_window_shard,
)
from v1b_metrics import (
    Reservoir,
    bootstrap_energy_ci,
    behavioral_window_distance,
    energy_distance,
    native_pair_distances,
    percentile_summary,
    spearman_rho,
    within_chunk_pair_distances,
)
from v1b_sampling import (
    condition_sampling_plan,
    deterministic_pairs,
    deterministic_sample_indices,
)


LEVELS = ("Z", "S", "S_tilde", "g")
R3_CONDITIONS = {
    "N1": "val",
    "A2": "actor2_test",
    "N2": "actor1_test",
}


@dataclass
class V1BConfig:
    artifact_root: Path
    window_root: Path
    output_root: Path
    dataset: str = "CU"
    seed: int = 42
    window_size: int = 30
    num_devices: int = 23
    representation_dim: int = 64
    r2_pair_count: int = 100_000
    r3_max_per_condition: int = 50_000
    r1_reservoir_size: int = 100_000
    r1_chunk_size: int = 512
    r3_block_length_windows: int = 30
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 42


class V1BEngine:
    def __init__(self, cfg: V1BConfig):
        self.cfg = cfg

    def _shape_gate(self, reps: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        expected_tail = {
            "Z": (23, 64),
            "S": (9, 64),
            "S_tilde": (9, 64),
            "g": (64,),
        }
        for level in LEVELS:
            if level not in reps:
                return False, f"missing_{level}"
            a = reps[level]
            if not np.isfinite(a).all():
                return False, f"nonfinite_{level}"
            if tuple(a.shape[1:]) != expected_tail[level]:
                return False, f"shape_{level}_{a.shape}"
            if a.dtype not in (np.float32, np.float64):
                return False, f"dtype_{level}_{a.dtype}"
        n = reps["g"].shape[0]
        if any(reps[l].shape[0] != n for l in LEVELS):
            return False, "cross_level_sample_count_mismatch"
        return True, "ok"

    def _collect_split_representation(
        self,
        split: str,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """Load a split into bounded output arrays.

        This is intended only for <=50k R3 samples. It is not used for the
        complete CU corpus. Full R1 processing remains shard-wise.
        """
        shard_ids = discover_shards(self.cfg.artifact_root, split)
        if not shard_ids:
            raise RuntimeError(f"No representation shards for split={split}")

        chunks = {l: [] for l in LEVELS}
        total = 0
        for sid in shard_ids:
            reps = load_representation_shard(
                self.cfg.artifact_root, split, sid
            )
            ok, reason = self._shape_gate(reps)
            if not ok:
                raise RuntimeError(f"Eligibility failed: {split}/{sid}: {reason}")
            for l in LEVELS:
                chunks[l].append(reps[l])
            total += reps["g"].shape[0]

        return (
            {l: np.concatenate(chunks[l], axis=0) for l in LEVELS},
            {"split": split, "sample_count": total, "shards": shard_ids},
        )

    def run_r1(self) -> dict:
        """R1 bounded streaming geometry summaries.

        Each shard is processed independently. Distances are computed only
        within bounded chunks. A deterministic reservoir prevents global
        distance materialization.
        """
        reservoirs = {
            l: Reservoir(self.cfg.r1_reservoir_size, self.cfg.seed + i)
            for i, l in enumerate(LEVELS)
        }
        shard_counts = {}
        eligible = 0

        # R1 is applied to non-training V1 populations.
        splits = ("val", "actor2_test", "actor1_test")

        for split in splits:
            for sid in discover_shards(self.cfg.artifact_root, split):
                reps = load_representation_shard(
                    self.cfg.artifact_root, split, sid
                )
                ok, reason = self._shape_gate(reps)
                if not ok:
                    shard_counts[f"{split}/{sid:06d}"] = {
                        "eligible": False,
                        "reason": reason,
                    }
                    continue

                eligible += 1
                n = reps["g"].shape[0]
                shard_counts[f"{split}/{sid:06d}"] = {
                    "eligible": True,
                    "sample_count": n,
                }

                for level in LEVELS:
                    x = reps[level]
                    for start in range(0, n, self.cfg.r1_chunk_size):
                        stop = min(start + self.cfg.r1_chunk_size, n)
                        vals = within_chunk_pair_distances(x[start:stop])
                        reservoirs[level].update(vals)

        summaries = {
            l: percentile_summary(reservoirs[l].values())
            for l in LEVELS
        }

        return {
            "measurement_family": "within_level_representation_geometry",
            "scope": "val + actor2_test + actor1_test",
            "reservoir_capacity": self.cfg.r1_reservoir_size,
            "chunk_size": self.cfg.r1_chunk_size,
            "levels": summaries,
            "eligible_shards": eligible,
            "shard_audit": shard_counts,
            "interpretation": "evidence_only_no_threshold",
        }

    def _load_r2_pairs(self):
        """Prepare deterministic global pairs over the V1 R2 population.

        This implementation uses the concatenated bounded R2 population only
        for the 100k-pair computation. It is NOT suitable for unbounded
        populations and must be replaced with a shard-index lookup if the
        combined window memory exceeds the project's safe budget.
        """
        splits = ("val", "actor2_test", "actor1_test")
        reps_all = {l: [] for l in LEVELS}
        windows_all = []
        provenance = []

        for split in splits:
            reps, meta = self._collect_split_representation(split)
            for l in LEVELS:
                reps_all[l].append(reps[l])

            for sid in discover_shards(self.cfg.artifact_root, split):
                w = load_window_shard(self.cfg.window_root, split, sid)
                x = w.get("X", w.get("windows"))
                if x is None:
                    raise KeyError(f"No X/windows array in {split}/{sid}")
                windows_all.append(np.asarray(x))
                provenance.extend([(split, sid)] * len(x))

        reps = {l: np.concatenate(reps_all[l], axis=0) for l in LEVELS}
        X = np.concatenate(windows_all, axis=0)
        pairs = deterministic_pairs(
            len(X),
            self.cfg.r2_pair_count,
            self.cfg.seed,
        )
        return X, reps, pairs, provenance

    def run_r2(self) -> dict:
        X, reps, pairs, provenance = self._load_r2_pairs()

        bx = np.empty(len(pairs), dtype=np.float64)
        dists = {l: np.empty(len(pairs), dtype=np.float64) for l in LEVELS}
        strata = {}

        for k, (i, j) in enumerate(pairs):
            bx[k] = behavioral_window_distance(X[i], X[j])
            for l in LEVELS:
                dists[l][k] = native_pair_distances(
                    reps[l][i:i+1], reps[l][j:j+1]
                )[0]

            si, sj = provenance[int(i)][0], provenance[int(j)][0]
            key = f"{si}-{sj}"
            strata[key] = strata.get(key, 0) + 1

        return {
            "measurement_family": "behavioral_relationship_preservation",
            "pair_count": int(len(pairs)),
            "seed": self.cfg.seed,
            "pair_sampling": "deterministic_representation_blind_without_replacement",
            "behavior_distance": {
                "definition": "frobenius_norm / (W*N)",
                "W": self.cfg.window_size,
                "N": self.cfg.num_devices,
            },
            "representation_distance": "within_native_level_euclidean",
            "spearman": {
                l: spearman_rho(bx, dists[l]) for l in LEVELS
            },
            "pair_strata": strata,
        }

    def run_r3(self) -> dict:
        populations = {}
        sampled = {}

        # First collect bounded R3 populations.
        for condition, split in R3_CONDITIONS.items():
            reps, meta = self._collect_split_representation(split)
            n = len(reps["g"])
            idx = deterministic_sample_indices(
                n,
                min(self.cfg.r3_max_per_condition, n),
                seed=self.cfg.seed + list(R3_CONDITIONS).index(condition),
            )
            sampled[condition] = {l: reps[l][idx] for l in LEVELS}
            populations[condition] = len(idx)

        comparisons = [
            ("N1", "A2"),
            ("A2", "N2"),
            ("N1", "N2"),
        ]

        result = {
            "measurement_family": "energy_distance",
            "sampling": {
                "max_per_condition": self.cfg.r3_max_per_condition,
                "without_replacement": True,
                "representation_blind": True,
                "seed": self.cfg.seed,
            },
            "conditions": {
                "N1": "val",
                "A2": "actor2_test",
                "N2": "actor1_test",
            },
            "sample_counts": populations,
            "comparisons": {},
            "uncertainty": {
                "method": "moving_block_bootstrap",
                "block_length_windows": self.cfg.r3_block_length_windows,
                "bootstrap_replicates": self.cfg.bootstrap_replicates,
                "confidence_level": 0.95,
                "interval_method": "percentile",
            },
        }

        for a, b in comparisons:
            key = f"{a}-{b}"
            result["comparisons"][key] = {}
            for l in LEVELS:
                x = sampled[a][l]
                y = sampled[b][l]

                # scipy's Energy distance is exact for the bounded sampled
                # populations. For very high-dimensional structured levels,
                # this can be computationally expensive; the integration
                # layer should benchmark before full execution.
                ci = bootstrap_energy_ci(
                    x.reshape(len(x), -1),
                    y.reshape(len(y), -1),
                    block_length=self.cfg.r3_block_length_windows,
                    replicates=self.cfg.bootstrap_replicates,
                    seed=self.cfg.bootstrap_seed,
                )
                result["comparisons"][key][l] = ci

        return result

    def run(self) -> dict:
        started = time.time()

        evidence = {
            "manifest": {
                "engine": "HDEG V1-B Scientific Validation Engine",
                "version": "1.0",
                "dataset": self.cfg.dataset,
                "seed": self.cfg.seed,
                "window_size": self.cfg.window_size,
                "num_devices": self.cfg.num_devices,
                "representation_dim": self.cfg.representation_dim,
                "r2_pair_count": self.cfg.r2_pair_count,
                "r3_max_per_condition": self.cfg.r3_max_per_condition,
                "r3_block_length_windows": self.cfg.r3_block_length_windows,
                "claim_decision": "not_performed",
            },
            "r1": self.run_r1(),
            "r2": self.run_r2(),
            "r3": self.run_r3(),
            "provenance": {
                "execution_started_unix": started,
                "execution_finished_unix": time.time(),
            },
        }
        return evidence
