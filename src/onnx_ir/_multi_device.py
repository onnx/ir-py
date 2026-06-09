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
    "check_device_configurations",
    "deserialize_model_configuration",
    "deserialize_node_device_configuration",
    "serialize_model_configuration",
    "serialize_node_device_configuration",
]

import dataclasses
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import onnx  # noqa: TID251

from onnx_ir._core import SymbolicDim, Value

if TYPE_CHECKING:
    from onnx_ir import _core

logger = logging.getLogger(__name__)


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


def deserialize_model_configuration(
    proto: onnx.DeviceConfigurationProto,
) -> ModelConfiguration:
    return ModelConfiguration(
        name=proto.name,
        num_devices=proto.num_devices,
        device=tuple(proto.device),
    )


def serialize_model_configuration(
    configuration: ModelConfiguration,
) -> onnx.DeviceConfigurationProto:
    proto = onnx.DeviceConfigurationProto()
    proto.name = configuration.name
    proto.num_devices = configuration.num_devices
    proto.device.extend(configuration.device)
    return proto


def _deserialize_simple_sharded_dim(proto: onnx.SimpleShardedDimProto) -> SimpleShardedDim:
    if proto.HasField("dim_value"):
        dim: int | SymbolicDim | None = proto.dim_value
    elif proto.HasField("dim_param"):
        dim = SymbolicDim(proto.dim_param)
    else:
        dim = None
    return SimpleShardedDim(
        dim=dim,
        num_shards=proto.num_shards,
    )


def _serialize_simple_sharded_dim(
    simple_sharding: SimpleShardedDim,
) -> onnx.SimpleShardedDimProto:
    proto = onnx.SimpleShardedDimProto()
    if isinstance(simple_sharding.dim, int):
        proto.dim_value = simple_sharding.dim
    elif (
        isinstance(simple_sharding.dim, SymbolicDim) and simple_sharding.dim.value is not None
    ):
        proto.dim_param = simple_sharding.dim.value
    proto.num_shards = simple_sharding.num_shards
    return proto


def _deserialize_sharded_dim(proto: onnx.ShardedDimProto) -> ShardedDim:
    return ShardedDim(
        axis=proto.axis,
        simple_sharding=tuple(
            _deserialize_simple_sharded_dim(simple_sharding)
            for simple_sharding in proto.simple_sharding
        ),
    )


def _serialize_sharded_dim(sharded_dim: ShardedDim) -> onnx.ShardedDimProto:
    proto = onnx.ShardedDimProto()
    proto.axis = sharded_dim.axis
    for simple_sharding in sharded_dim.simple_sharding:
        proto.simple_sharding.append(_serialize_simple_sharded_dim(simple_sharding))
    return proto


def _resolve_value(tensor_name: str, values: Mapping[str, _core.Value]) -> _core.Value | None:
    """Resolve a proto ``tensor_name`` to a :class:`~onnx_ir.Value`.

    When the name is not present in ``values`` a placeholder value carrying the
    name is created so the reference round-trips losslessly. This mirrors how
    missing node inputs are handled during deserialization.
    """
    if not tensor_name:
        return None
    value = values.get(tensor_name)
    if value is None:
        logger.warning(
            "ShardingSpec references tensor '%s' which is not found in the current "
            "scope. Creating a placeholder value.",
            tensor_name,
        )
        value = Value(name=tensor_name)
    return value


def _deserialize_sharding_spec(
    proto: onnx.ShardingSpecProto, values: Mapping[str, _core.Value]
) -> ShardingSpec:
    return ShardingSpec(
        value=_resolve_value(proto.tensor_name, values),
        device=tuple(proto.device),
        index_to_device_group_map=tuple(
            IndexToDeviceGroupMapEntry(key=entry.key, value=tuple(entry.value))
            for entry in proto.index_to_device_group_map
        ),
        sharded_dim=tuple(_deserialize_sharded_dim(dim) for dim in proto.sharded_dim),
    )


def _serialize_sharding_spec(sharding_spec: ShardingSpec) -> onnx.ShardingSpecProto:
    proto = onnx.ShardingSpecProto()
    if sharding_spec.value is not None:
        name = sharding_spec.value.name
        if not name:
            raise ValueError(
                "Cannot serialize a ShardingSpec whose value has no name. "
                f"Value: {sharding_spec.value!r}"
            )
        proto.tensor_name = name
    proto.device.extend(sharding_spec.device)
    for entry in sharding_spec.index_to_device_group_map:
        map_entry = proto.index_to_device_group_map.add()
        map_entry.key = entry.key
        map_entry.value.extend(entry.value)
    for sharded_dim in sharding_spec.sharded_dim:
        proto.sharded_dim.append(_serialize_sharded_dim(sharded_dim))
    return proto


def deserialize_node_device_configuration(
    proto: onnx.NodeDeviceConfigurationProto,
    values: Mapping[str, _core.Value] | None = None,
) -> NodeDeviceConfiguration:
    """Deserialize a node device configuration.

    Args:
        proto: The proto to deserialize.
        values: Mapping from value name to :class:`~onnx_ir.Value` used to
            resolve ``tensor_name`` references. When ``None`` all references
            resolve to freshly created placeholder values.

    Note:
        ``configuration`` is resolved to a *placeholder* :class:`ModelConfiguration`
        carrying the ``configuration_id``. The placeholder is replaced by the real
        model configuration object by a post-pass once the model is fully built
        (see :func:`onnx_ir.serde.deserialize_model`).
    """
    if values is None:
        values = {}
    configuration: ModelConfiguration | None = None
    if proto.configuration_id:
        configuration = ModelConfiguration(name=proto.configuration_id, num_devices=0)
    pipeline_stage = proto.pipeline_stage if proto.HasField("pipeline_stage") else None
    return NodeDeviceConfiguration(
        configuration=configuration,
        sharding_spec=tuple(
            _deserialize_sharding_spec(spec, values) for spec in proto.sharding_spec
        ),
        pipeline_stage=pipeline_stage,
    )


def serialize_node_device_configuration(
    node_device_configuration: NodeDeviceConfiguration,
) -> onnx.NodeDeviceConfigurationProto:
    proto = onnx.NodeDeviceConfigurationProto()
    if node_device_configuration.configuration is not None:
        name = node_device_configuration.configuration.name
        if not name:
            raise ValueError(
                "Cannot serialize a NodeDeviceConfiguration whose configuration has "
                f"no name. Configuration: {node_device_configuration.configuration!r}"
            )
        proto.configuration_id = name
    for sharding_spec in node_device_configuration.sharding_spec:
        proto.sharding_spec.append(_serialize_sharding_spec(sharding_spec))
    if node_device_configuration.pipeline_stage is not None:
        proto.pipeline_stage = node_device_configuration.pipeline_stage
    return proto


def check_device_configurations(model: _core.Model) -> list[str]:
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
        for device_index in spec.device:
            if device_index < 0 or device_index >= num_devices:
                errors.append(
                    f"Node '{label}': device index {device_index} for value "
                    f"'{spec.value.name}' is out of range (num_devices={num_devices})."
                )
