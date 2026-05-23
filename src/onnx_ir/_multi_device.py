# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

__all__ = [
    "IndexToDeviceGroupMapEntry",
    "ModelConfiguration",
    "NodeDeviceConfiguration",
    "ShardedDim",
    "ShardingSpec",
    "SimpleShardedDim",
    "deserialize_model_configuration",
    "deserialize_node_device_configuration",
    "serialize_model_configuration",
    "serialize_node_device_configuration",
]

import dataclasses

import onnx  # noqa: TID251

from onnx_ir._core import SymbolicDim


@dataclasses.dataclass(frozen=True)
class ModelConfiguration:
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
    tensor_name: str
    device: tuple[int, ...] = ()
    index_to_device_group_map: tuple[IndexToDeviceGroupMapEntry, ...] = ()
    sharded_dim: tuple[ShardedDim, ...] = ()


@dataclasses.dataclass(frozen=True)
class NodeDeviceConfiguration:
    configuration_id: str
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


def _deserialize_sharding_spec(proto: onnx.ShardingSpecProto) -> ShardingSpec:
    return ShardingSpec(
        tensor_name=proto.tensor_name,
        device=tuple(proto.device),
        index_to_device_group_map=tuple(
            IndexToDeviceGroupMapEntry(key=entry.key, value=tuple(entry.value))
            for entry in proto.index_to_device_group_map
        ),
        sharded_dim=tuple(_deserialize_sharded_dim(dim) for dim in proto.sharded_dim),
    )


def _serialize_sharding_spec(sharding_spec: ShardingSpec) -> onnx.ShardingSpecProto:
    proto = onnx.ShardingSpecProto()
    proto.tensor_name = sharding_spec.tensor_name
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
) -> NodeDeviceConfiguration:
    pipeline_stage = proto.pipeline_stage if proto.HasField("pipeline_stage") else None
    return NodeDeviceConfiguration(
        configuration_id=proto.configuration_id,
        sharding_spec=tuple(_deserialize_sharding_spec(spec) for spec in proto.sharding_spec),
        pipeline_stage=pipeline_stage,
    )


def serialize_node_device_configuration(
    node_device_configuration: NodeDeviceConfiguration,
) -> onnx.NodeDeviceConfigurationProto:
    proto = onnx.NodeDeviceConfigurationProto()
    proto.configuration_id = node_device_configuration.configuration_id
    for sharding_spec in node_device_configuration.sharding_spec:
        proto.sharding_spec.append(_serialize_sharding_spec(sharding_spec))
    if node_device_configuration.pipeline_stage is not None:
        proto.pipeline_stage = node_device_configuration.pipeline_stage
    return proto
