# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Object-bound multi-device (IRv11) configuration metadata.

Unlike the ONNX protos, which reference tensors and device configurations by
*name*, the IR representation binds directly to the :class:`~onnx_ir.Value` and
:class:`ModelConfiguration` *objects*. The proto-level ``tensor_name`` and
``configuration_id`` strings are therefore **derived** from ``value.name`` and
``configuration.name`` at serialization time, so there is a single source of
truth and references follow renames automatically.
"""

from __future__ import annotations

__all__ = [
    "IndexToDeviceGroupMapEntry",
    "ModelConfiguration",
    "NodeDeviceConfiguration",
    "ShardedDim",
    "ShardingSpec",
    "SimpleShardedDim",
]

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx_ir import _core
    from onnx_ir._core import SymbolicDim, Value


@dataclasses.dataclass(frozen=True)
class ModelConfiguration:
    """A model-level device configuration (mirrors ``DeviceConfigurationProto``)."""

    name: str
    num_devices: int
    device: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class IndexToDeviceGroupMapEntry:
    key: int
    value: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class SimpleShardedDim:
    """A dimension with a sharding specification.

    The ``dim`` field represents the size of the dimension and follows the same
    convention as :class:`onnx_ir.Shape` dimensions: an :class:`int` for a
    fixed size, a :class:`~onnx_ir.SymbolicDim` for a symbolic/unknown size,
    or ``None`` when the dimension is unspecified.
    """

    dim: int | SymbolicDim | None = None
    num_shards: int = 0


@dataclasses.dataclass(frozen=True)
class ShardedDim:
    axis: int
    simple_sharding: tuple[SimpleShardedDim, ...] = ()


@dataclasses.dataclass(frozen=True)
class ShardingSpec:
    """Sharding for a single tensor.

    ``value`` binds directly to the :class:`~onnx_ir.Value` being sharded. The
    proto ``tensor_name`` is derived from ``value.name`` on serialization.
    """

    value: Value | None = None
    device: tuple[int, ...] = ()
    index_to_device_group_map: tuple[IndexToDeviceGroupMapEntry, ...] = ()
    sharded_dim: tuple[ShardedDim, ...] = ()


@dataclasses.dataclass(frozen=True)
class NodeDeviceConfiguration:
    """Per-node device configuration.

    ``configuration`` binds directly to the :class:`ModelConfiguration` object.
    The proto ``configuration_id`` is derived from ``configuration.name`` on
    serialization.
    """

    configuration: ModelConfiguration | None = None
    sharding_spec: tuple[ShardingSpec, ...] = ()
    pipeline_stage: int | None = None


def _check_device_configurations(model: _core.Model) -> list[str]:
    """Validate the multi-device invariants of a model.

    Returns a list of human-readable violation messages. An empty list means the
    model satisfies all invariants. This never raises for invariant violations;
    callers decide whether to warn or raise.

    Checks:
        * INV-2 reachability: every ``ShardingSpec.value`` is an input/output of
          its node and every ``NodeDeviceConfiguration.configuration`` exists in
          ``model.device_configurations``.
        * INV-3 nameability: referenced values and configurations have non-empty
          names (otherwise serialization fails).
        * INV-4 structural: ``ShardedDim.axis`` is in range when the value rank is
          known, ``num_shards >= 1``, and device indices are within ``num_devices``.
    """
    errors: list[str] = []
    known_configs = {config.name: config for config in model.device_configurations}

    def node_label(node: _core.Node) -> str:
        return node.name or f"<anonymous {node.op_type}>"

    all_nodes = list(model.graph.all_nodes())
    for func in model.functions.values():
        all_nodes.extend(func.all_nodes())

    for node in all_nodes:
        device_configurations = node.device_configurations
        if not device_configurations:
            continue
        node_io = set(node.inputs) | set(node.outputs)
        for config in device_configurations:
            if config.configuration is None:
                errors.append(
                    f"Node '{node_label(node)}' has a device configuration without a "
                    "ModelConfiguration reference."
                )
            else:
                config_name = config.configuration.name
                if not config_name:
                    errors.append(
                        f"Node '{node_label(node)}' references a configuration with an "
                        "empty name (cannot be serialized)."
                    )
                elif config_name not in known_configs:
                    errors.append(
                        f"Node '{node_label(node)}' references configuration "
                        f"'{config_name}' which is not declared in "
                        "model.device_configurations."
                    )
            num_devices = (
                config.configuration.num_devices if config.configuration is not None else None
            )
            for spec in config.sharding_spec:
                _check_sharding_spec(spec, node, node_io, num_devices, errors)

    return errors


def _check_sharding_spec(
    spec: ShardingSpec,
    node: _core.Node,
    node_io: set,
    num_devices: int | None,
    errors: list[str],
) -> None:
    label = node.name or f"<anonymous {node.op_type}>"
    if spec.value is None:
        errors.append(f"Node '{label}' has a ShardingSpec without a value.")
        return
    if not spec.value.name:
        errors.append(
            f"Node '{label}' shards a value with an empty name (cannot be serialized)."
        )
    if spec.value not in node_io:
        errors.append(
            f"Node '{label}' shards value '{spec.value.name}' which is not an input "
            "or output of the node."
        )
    rank = len(spec.value.shape) if spec.value.shape is not None else None

    def normalize_axis(axis: int) -> int:
        return axis + rank if (rank is not None and axis < 0) else axis

    seen_axes: set[int] = set()
    for sharded_dim in spec.sharded_dim:
        # ONNX allows negative axes in the range [-rank, rank - 1].
        if rank is not None and not -rank <= sharded_dim.axis < rank:
            errors.append(
                f"Node '{label}': sharded axis {sharded_dim.axis} of value "
                f"'{spec.value.name}' is out of range (rank={rank})."
            )
        else:
            normalized = normalize_axis(sharded_dim.axis)
            if normalized in seen_axes:
                errors.append(
                    f"Node '{label}': value '{spec.value.name}' is sharded along "
                    f"axis {sharded_dim.axis} more than once."
                )
            seen_axes.add(normalized)
        for simple in sharded_dim.simple_sharding:
            if simple.num_shards < 1:
                errors.append(
                    f"Node '{label}': num_shards={simple.num_shards} for value "
                    f"'{spec.value.name}' must be >= 1."
                )
    if num_devices is not None:
        # An entry in ``device`` is either a direct device id in
        # [0, num_devices), or a key into ``index_to_device_group_map`` (often
        # negative) that names a group of real device ids — used to replicate a
        # shard across a set of devices. Validate accordingly.
        group_map = {entry.key: entry.value for entry in spec.index_to_device_group_map}
        for device_index in spec.device:
            if device_index in group_map:
                for member in group_map[device_index]:
                    if not 0 <= member < num_devices:
                        errors.append(
                            f"Node '{label}': device {member} in group "
                            f"{device_index} for value '{spec.value.name}' is out "
                            f"of range (num_devices={num_devices})."
                        )
            elif not 0 <= device_index < num_devices:
                errors.append(
                    f"Node '{label}': device index {device_index} for value "
                    f"'{spec.value.name}' is out of range (num_devices={num_devices})."
                )
