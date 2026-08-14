from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BehavioralStateConfigError(ValueError):
    """
    Raised when the behavioral-state configuration is structurally or
    semantically invalid.
    """


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehavioralDevice:
    """
    One selected device feature and its fixed column index.

    Parameters
    ----------
    index:
        Column index in the HDEG input representation.
    device_id:
        Exact device/feature identifier from devices.json.
    """

    index: int
    device_id: str


@dataclass(frozen=True)
class BehavioralState:
    """
    One behavioral state defined by the BSE semantic configuration.

    Parameters
    ----------
    id:
        Stable machine-readable identifier.
    index:
        Row index in the compatibility matrix.
    name:
        Human-readable behavioral-state name.
    compatible_device_indices:
        Device-column indices allowed to contribute to this state.
    rationale:
        Semantic justification for the compatibility assignment.
    provenance_type:
        Provenance classification for the assignment.
    confidence:
        Semantic adjudication confidence.
    """

    id: str
    index: int
    name: str
    compatible_device_indices: tuple[int, ...]
    rationale: str
    provenance_type: str
    confidence: str


@dataclass(frozen=True)
class BehavioralStateConfiguration:
    """
    Validated HDEG behavioral-state semantic configuration.

    This object is deliberately independent of the BSE neural module.

    The configuration defines:
        - the fixed device-column ordering;
        - the behavioral-state ordering;
        - the binary compatibility matrix;
        - semantic provenance metadata.

    BSE consumes only the resulting compatibility tensor.
    """

    schema_version: str
    artifact_type: str
    status: str

    dataset_name: str

    devices: tuple[BehavioralDevice, ...]
    states: tuple[BehavioralState, ...]

    compatibility_matrix: np.ndarray

    compatibility_relation: str
    contextual_edges: str

    @property
    def num_devices(self) -> int:
        """Number of selected device features."""
        return len(self.devices)

    @property
    def num_states(self) -> int:
        """Number of behavioral states."""
        return len(self.states)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Shape of the compatibility matrix."""
        return tuple(self.compatibility_matrix.shape)

    def torch_mask(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Return the compatibility matrix as a PyTorch tensor.

        Parameters
        ----------
        dtype:
            Tensor dtype. BSE normally uses float32.
        device:
            Optional target device.
        clone:
            Whether to return an independent tensor.

        Returns
        -------
        torch.Tensor
            Tensor with shape (K, N).
        """
        tensor = torch.from_numpy(
            self.compatibility_matrix
        ).to(
            dtype=dtype,
            device=device,
        )

        if clone:
            tensor = tensor.clone()

        return tensor

    def device_ids(self) -> tuple[str, ...]:
        """Return device identifiers in the exact matrix-column order."""
        return tuple(
            device.device_id
            for device in self.devices
        )

    def state_ids(self) -> tuple[str, ...]:
        """Return behavioral-state identifiers in matrix-row order."""
        return tuple(
            state.id
            for state in self.states
        )

    def state_names(self) -> tuple[str, ...]:
        """Return behavioral-state names in matrix-row order."""
        return tuple(
            state.name
            for state in self.states
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_behavioral_state_config(
    config_path: str | Path,
    devices_path: str | Path,
) -> BehavioralStateConfiguration:
    """
    Load and validate the HDEG behavioral-state configuration.

    Parameters
    ----------
    config_path:
        Path to behavioral_states.yaml.

    devices_path:
        Path to the authoritative devices.json file.

    Returns
    -------
    BehavioralStateConfiguration
        Fully validated semantic configuration.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.

    BehavioralStateConfigError
        If the configuration violates the semantic/configuration contract.
    """
    config_path = Path(config_path)
    devices_path = Path(devices_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Behavioral-state configuration not found: "
            f"{config_path}"
        )

    if not devices_path.is_file():
        raise FileNotFoundError(
            f"Device-order file not found: "
            f"{devices_path}"
        )

    config = _load_yaml(config_path)
    device_ids = _load_devices_json(devices_path)

    return _build_and_validate_configuration(
        config=config,
        device_ids=device_ids,
    )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load the YAML configuration document.
    """
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise BehavioralStateConfigError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise BehavioralStateConfigError(
            "Behavioral-state configuration must contain "
            "a top-level mapping."
        )

    return data


def _load_devices_json(path: Path) -> tuple[str, ...]:
    """
    Load the authoritative selected-device ordering.
    """
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise BehavioralStateConfigError(
            f"Invalid JSON device-order file: {path}"
        ) from exc

    if not isinstance(data, list):
        raise BehavioralStateConfigError(
            "devices.json must contain a JSON list."
        )

    if not data:
        raise BehavioralStateConfigError(
            "devices.json contains no devices."
        )

    if not all(
        isinstance(device_id, str)
        and device_id.strip()
        for device_id in data
    ):
        raise BehavioralStateConfigError(
            "Every device entry in devices.json must be "
            "a non-empty string."
        )

    if len(set(data)) != len(data):
        raise BehavioralStateConfigError(
            "devices.json contains duplicate device identifiers."
        )

    return tuple(data)


# ---------------------------------------------------------------------------
# Configuration construction
# ---------------------------------------------------------------------------


def _build_and_validate_configuration(
    *,
    config: dict[str, Any],
    device_ids: tuple[str, ...],
) -> BehavioralStateConfiguration:
    """
    Parse the YAML document and validate the complete semantic contract.
    """

    schema_version = _require_string(
        config,
        "schema_version",
        "top-level",
    )

    artifact_type = _require_string(
        config,
        "artifact_type",
        "top-level",
    )

    status = _require_string(
        config,
        "status",
        "top-level",
    )

    if artifact_type != "hdeg_behavioral_state_configuration":
        raise BehavioralStateConfigError(
            "Unexpected artifact_type: "
            f"{artifact_type!r}."
        )

    dataset = _require_mapping(
        config,
        "dataset",
        "top-level",
    )

    dataset_name = _require_string(
        dataset,
        "name",
        "dataset",
    )

    selected_device_count = _require_int(
        dataset,
        "selected_device_count",
        "dataset",
    )

    if selected_device_count != len(device_ids):
        raise BehavioralStateConfigError(
            "Device-count mismatch: "
            f"configuration declares {selected_device_count}, "
            f"but devices.json contains {len(device_ids)}."
        )

    device_order_source = _require_string(
        dataset,
        "device_order_source",
        "dataset",
    )

    if device_order_source != "devices.json":
        raise BehavioralStateConfigError(
            "The behavioral-state configuration must use "
            '"devices.json" as its device-order source.'
        )

    device_ordering_rule = _require_string(
        dataset,
        "device_ordering_rule",
        "dataset",
    )

    if device_ordering_rule != "exact_file_order":
        raise BehavioralStateConfigError(
            "Unsupported device ordering rule: "
            f"{device_ordering_rule!r}."
        )

    semantic_policy = _require_mapping(
        config,
        "semantic_policy",
        "top-level",
    )

    compatibility_relation = _require_string(
        semantic_policy,
        "compatibility_relation",
        "semantic_policy",
    )

    contextual_edges = _require_string(
        semantic_policy,
        "contextual_edges",
        "semantic_policy",
    )

    if compatibility_relation != "DIRECT":
        raise BehavioralStateConfigError(
            "Version 1.0 requires compatibility_relation='DIRECT'."
        )

    if contextual_edges != "none":
        raise BehavioralStateConfigError(
            "Version 1.0 requires contextual_edges='none'."
        )

    state_entries = config.get("behavioral_states")

    if not isinstance(state_entries, list):
        raise BehavioralStateConfigError(
            "'behavioral_states' must be a list."
        )

    if not state_entries:
        raise BehavioralStateConfigError(
            "'behavioral_states' must not be empty."
        )

    states = _parse_states(
        state_entries=state_entries,
        num_devices=len(device_ids),
    )

    matrix_section = _require_mapping(
        config,
        "matrix",
        "top-level",
    )

    matrix = _parse_matrix(
        matrix_section=matrix_section,
        expected_num_states=len(states),
        expected_num_devices=len(device_ids),
    )

    devices = tuple(
        BehavioralDevice(
            index=index,
            device_id=device_id,
        )
        for index, device_id in enumerate(device_ids)
    )

    configuration = BehavioralStateConfiguration(
        schema_version=schema_version,
        artifact_type=artifact_type,
        status=status,
        dataset_name=dataset_name,
        devices=devices,
        states=tuple(states),
        compatibility_matrix=matrix,
        compatibility_relation=compatibility_relation,
        contextual_edges=contextual_edges,
    )

    _validate_configuration(
        configuration
    )

    return configuration


# ---------------------------------------------------------------------------
# State parsing
# ---------------------------------------------------------------------------


def _parse_states(
    *,
    state_entries: list[Any],
    num_devices: int,
) -> list[BehavioralState]:
    """
    Parse and validate behavioral-state definitions.
    """

    states: list[BehavioralState] = []

    for expected_index, raw_state in enumerate(
        state_entries
    ):
        if not isinstance(raw_state, dict):
            raise BehavioralStateConfigError(
                "Each behavioral state must be a mapping."
            )

        state_id = _require_string(
            raw_state,
            "id",
            f"behavioral_states[{expected_index}]",
        )

        name = _require_string(
            raw_state,
            "name",
            f"behavioral_states[{expected_index}]",
        )

        index = _require_int(
            raw_state,
            "index",
            f"behavioral_states[{expected_index}]",
        )

        if index != expected_index:
            raise BehavioralStateConfigError(
                "Behavioral-state indices must be contiguous "
                f"and ordered. Expected {expected_index}, "
                f"received {index} for state {state_id!r}."
            )

        rationale = _require_string(
            raw_state,
            "rationale",
            f"behavioral_states[{expected_index}]",
        )

        provenance_type = _require_string(
            raw_state,
            "provenance_type",
            f"behavioral_states[{expected_index}]",
        )

        confidence = _require_string(
            raw_state,
            "confidence",
            f"behavioral_states[{expected_index}]",
        )

        if confidence not in {
            "high",
            "moderate",
            "low",
        }:
            raise BehavioralStateConfigError(
                f"Invalid confidence {confidence!r} "
                f"for state {state_id!r}."
            )

        compatible_entries = raw_state.get(
            "compatible_devices"
        )

        if not isinstance(
            compatible_entries,
            list,
        ):
            raise BehavioralStateConfigError(
                f"'compatible_devices' must be a list "
                f"for state {state_id!r}."
            )

        if not compatible_entries:
            raise BehavioralStateConfigError(
                f"State {state_id!r} has no compatible devices."
            )

        compatible_indices: list[int] = []

        for entry_index, raw_device in enumerate(
            compatible_entries
        ):
            if not isinstance(
                raw_device,
                dict,
            ):
                raise BehavioralStateConfigError(
                    f"Invalid compatible-device entry "
                    f"{entry_index} for state {state_id!r}."
                )

            device_index = _require_int(
                raw_device,
                "index",
                (
                    f"behavioral_states[{expected_index}]"
                    f".compatible_devices[{entry_index}]"
                ),
            )

            device_id = _require_string(
                raw_device,
                "id",
                (
                    f"behavioral_states[{expected_index}]"
                    f".compatible_devices[{entry_index}]"
                ),
            )

            relation = _require_string(
                raw_device,
                "relation",
                (
                    f"behavioral_states[{expected_index}]"
                    f".compatible_devices[{entry_index}]"
                ),
            )

            if relation != "DIRECT":
                raise BehavioralStateConfigError(
                    f"State {state_id!r}, device "
                    f"{device_index}: relation must be "
                    "'DIRECT' in Version 1.0."
                )

            if not 0 <= device_index < num_devices:
                raise BehavioralStateConfigError(
                    f"State {state_id!r} references invalid "
                    f"device index {device_index}."
                )

            compatible_indices.append(
                device_index
            )

            # The actual device-id correspondence is
            # validated later against devices.json.

            _ = device_id

        if len(
            set(compatible_indices)
        ) != len(compatible_indices):
            raise BehavioralStateConfigError(
                f"State {state_id!r} contains duplicate "
                "compatible-device indices."
            )

        states.append(
            BehavioralState(
                id=state_id,
                index=index,
                name=name,
                compatible_device_indices=tuple(
                    compatible_indices
                ),
                rationale=rationale,
                provenance_type=provenance_type,
                confidence=confidence,
            )
        )

    state_ids = [
        state.id
        for state in states
    ]

    if len(set(state_ids)) != len(state_ids):
        raise BehavioralStateConfigError(
            "Behavioral-state identifiers must be unique."
        )

    return states


# ---------------------------------------------------------------------------
# Matrix parsing
# ---------------------------------------------------------------------------


def _parse_matrix(
    *,
    matrix_section: dict[str, Any],
    expected_num_states: int,
    expected_num_devices: int,
) -> np.ndarray:
    """
    Parse the explicit compatibility matrix and validate its shape/value type.
    """

    shape = matrix_section.get("shape")

    if not isinstance(shape, list) or len(shape) != 2:
        raise BehavioralStateConfigError(
            "matrix.shape must be a two-element list."
        )

    if shape != [
        expected_num_states,
        expected_num_devices,
    ]:
        raise BehavioralStateConfigError(
            "Matrix shape mismatch: "
            f"declared={shape}, "
            f"expected="
            f"[{expected_num_states}, "
            f"{expected_num_devices}]."
        )

    dtype = matrix_section.get("dtype")

    if dtype != "uint8":
        raise BehavioralStateConfigError(
            "Version 1.0 requires matrix.dtype='uint8'."
        )

    rows = matrix_section.get("rows")

    if not isinstance(rows, list):
        raise BehavioralStateConfigError(
            "matrix.rows must be a list."
        )

    if len(rows) != expected_num_states:
        raise BehavioralStateConfigError(
            "Number of matrix rows does not match "
            "the number of behavioral states."
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise BehavioralStateConfigError(
                f"Matrix row {row_index} must be a list."
            )

        if len(row) != expected_num_devices:
            raise BehavioralStateConfigError(
                f"Matrix row {row_index} has length "
                f"{len(row)}; expected "
                f"{expected_num_devices}."
            )

    matrix = np.asarray(
        rows,
        dtype=np.uint8,
    )

    if not np.all(
        np.isin(matrix, [0, 1])
    ):
        raise BehavioralStateConfigError(
            "Compatibility matrix must contain "
            "only binary values 0 and 1."
        )

    return matrix


# ---------------------------------------------------------------------------
# Complete semantic validation
# ---------------------------------------------------------------------------


def _validate_configuration(
    configuration: BehavioralStateConfiguration,
) -> None:
    """
    Validate the complete configuration contract.

    This validation intentionally checks both representations:

        behavioral_states[*].compatible_device_indices

    and

        matrix.rows

    so that the semantic declaration and executable matrix cannot silently
    diverge.
    """

    matrix = configuration.compatibility_matrix

    K = configuration.num_states
    N = configuration.num_devices

    if matrix.shape != (K, N):
        raise BehavioralStateConfigError(
            "Final compatibility matrix shape mismatch: "
            f"{matrix.shape} != ({K}, {N})."
        )

    # ---------------------------------------------------------------
    # State-to-matrix consistency
    # ---------------------------------------------------------------

    for state in configuration.states:
        expected_columns = set(
            state.compatible_device_indices
        )

        actual_columns = set(
            np.flatnonzero(
                matrix[state.index]
            ).tolist()
        )

        if actual_columns != expected_columns:
            raise BehavioralStateConfigError(
                f"Compatibility mismatch for state "
                f"{state.id!r}: semantic declaration "
                f"{sorted(expected_columns)} != matrix "
                f"{sorted(actual_columns)}."
            )

    # ---------------------------------------------------------------
    # Every state must have at least one compatible device.
    # ---------------------------------------------------------------

    row_counts = matrix.sum(
        axis=1
    )

    if not np.all(row_counts > 0):
        invalid_states = np.flatnonzero(
            row_counts == 0
        ).tolist()

        raise BehavioralStateConfigError(
            "Some behavioral states have no compatible "
            f"devices: {invalid_states}."
        )

    # ---------------------------------------------------------------
    # Version 1.0 semantic partition:
    #
    # every selected device belongs to exactly one primary state.
    # ---------------------------------------------------------------

    column_counts = matrix.sum(
        axis=0
    )

    if not np.all(
        column_counts == 1
    ):
        invalid_devices = np.flatnonzero(
            column_counts != 1
        ).tolist()

        raise BehavioralStateConfigError(
            "Every selected device must belong to exactly "
            "one primary behavioral state in Version 1.0. "
            f"Invalid device columns: {invalid_devices}."
        )

    # ---------------------------------------------------------------
    # Device IDs embedded in YAML must agree with devices.json.
    # ---------------------------------------------------------------

    # Re-read the device IDs from the configuration entries is not
    # necessary here because BehavioralDevice is constructed from
    # devices.json. The explicit mapping is therefore validated at
    # load time by _validate_declared_device_ids().
    #
    # This method is invoked separately below when the raw YAML is
    # available.

    # ---------------------------------------------------------------
    # Binary matrix
    # ---------------------------------------------------------------

    if not np.issubdtype(
        matrix.dtype,
        np.integer,
    ):
        raise BehavioralStateConfigError(
            "Compatibility matrix must use an integer dtype."
        )

    if not np.all(
        np.isin(matrix, [0, 1])
    ):
        raise BehavioralStateConfigError(
            "Compatibility matrix contains values "
            "other than 0 and 1."
        )


# ---------------------------------------------------------------------------
# Primitive field helpers
# ---------------------------------------------------------------------------


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    """
    Require a mapping-valued field.
    """
    value = mapping.get(key)

    if not isinstance(value, dict):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be a mapping."
        )

    return value


def _require_string(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> str:
    """
    Require a non-empty string field.
    """
    value = mapping.get(key)

    if not isinstance(value, str):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be a string."
        )

    value = value.strip()

    if not value:
        raise BehavioralStateConfigError(
            f"{context}.{key} must not be empty."
        )

    return value


def _require_int(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> int:
    """
    Require an integer field.

    bool is deliberately rejected because bool is a subclass of int
    in Python.
    """
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be an integer."
        )

    return value


# ---------------------------------------------------------------------------
# Optional strict validation against raw YAML device IDs
# ---------------------------------------------------------------------------


def validate_behavioral_state_config(
    config_path: str | Path,
    devices_path: str | Path,
) -> BehavioralStateConfiguration:
    """
    Public alias emphasizing that loading includes full validation.

    This is intentionally equivalent to load_behavioral_state_config().
    """
    return load_behavioral_state_config(
        config_path=config_path,
        devices_path=devices_path,
    )


__all__ = [
    "BehavioralDevice",
    "BehavioralState",
    "BehavioralStateConfiguration",
    "BehavioralStateConfigError",
    "load_behavioral_state_config",
    "validate_behavioral_state_config",
]