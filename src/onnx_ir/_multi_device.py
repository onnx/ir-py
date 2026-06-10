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
    """A model-level set of devices that a multi-device plan runs on.

    A model declares one or more configurations (mirroring ONNX's
    ``DeviceConfigurationProto``). Each one names a group of devices; the
    per-node :class:`NodeDeviceConfiguration` entries then reference a
    configuration by object identity to say how that node is sharded or staged
    across those devices.

    Device *indices* used elsewhere (for example :attr:`ShardingSpec.device` and
    :meth:`onnx_ir.Node.shard`) are 0-based positions into :attr:`device`.
    """

    name: str
    """Unique name of the configuration within the model.

    Used as the ``configuration_id`` that node configurations refer to, so it
    must be non-empty to serialize.
    """

    num_devices: int
    """Number of devices in this configuration.

    Device indices are expected to fall in ``range(num_devices)``.
    """

    device: tuple[str, ...] = ()
    """Optional human-readable device names, one per device.

    For example ``("CPU", "CUDA:0")`` or ``("NPU", "GPU")``. When provided it
    should have length :attr:`num_devices`, and ``device[i]`` is the name of the
    device referenced by index ``i``. An empty tuple means the names are
    unspecified (the devices are then known only by index).
    """


@dataclasses.dataclass(frozen=True)
class IndexToDeviceGroupMapEntry:
    """One entry of a :attr:`ShardingSpec.index_to_device_group_map`.

    It maps a single key appearing in :attr:`ShardingSpec.device` to the group of
    real device indices that key stands for. This is how a shard is *replicated*
    across several devices: instead of naming one device, an entry in ``device``
    names a group, and this map expands that group.
    """

    key: int
    """The value that appears in :attr:`ShardingSpec.device` and is expanded here.

    Group keys are conventionally negative (e.g. ``-1``, ``-2``) so they cannot be
    confused with direct device indices, which are ``>= 0``.
    """

    value: tuple[int, ...] = ()
    """The real device indices (into :attr:`ModelConfiguration.device`) in the group.

    The shard assigned to :attr:`key` is replicated onto every device listed here.
    For example ``value=(0, 1)`` replicates the shard onto devices 0 and 1.
    """


@dataclasses.dataclass(frozen=True)
class SimpleShardedDim:
    """How one axis is divided into shards (``N`` blocks split into ``num_shards``)."""

    dim: int | SymbolicDim | None = None
    """Size of the axis being sharded.

    Follows the same convention as :class:`onnx_ir.Shape` dimensions:

    * an :class:`int` for a fixed, known size (e.g. ``1024``);
    * a :class:`~onnx_ir.SymbolicDim` for a symbolic/unknown size (e.g. a dynamic
      ``"batch"`` dimension);
    * ``None`` when the size is unspecified.

    ONNX permits the size to be symbolic, but :attr:`num_shards` must be a concrete
    integer.
    """

    num_shards: int = 0
    """Number of equal shards the axis is split into.

    ``1`` means the axis is not split (a single shard); a value ``>= 1`` is
    required. For example ``dim=1024, num_shards=4`` splits the axis into four
    blocks of 256.
    """


@dataclasses.dataclass(frozen=True)
class ShardedDim:
    """The sharding of a single tensor axis."""

    axis: int
    """The tensor axis this sharding applies to.

    Follows the ONNX axis convention: it must be in ``[-rank, rank - 1]``, where
    negative values count from the back (``-1`` is the last axis).
    """

    simple_sharding: tuple[SimpleShardedDim, ...] = ()
    """How the axis is divided, as a tuple of :class:`SimpleShardedDim`.

    The common case is a single entry. Multiple entries are used when a reshape
    has fused several original axes into this one, describing the sharding of each
    fused component in order.
    """


@dataclasses.dataclass(frozen=True)
class ShardingSpec:
    """How a single tensor is sharded and/or replicated across devices.

    A spec describes the distribution of one input or output tensor of a node.
    The unsharded axes are implicitly replicated; :attr:`sharded_dim` lists only
    the axes that are split.
    """

    value: Value | None = None
    """The :class:`~onnx_ir.Value` being sharded, bound by object identity.

    The proto ``tensor_name`` is derived from ``value.name`` on serialization, so
    the reference follows renames. It must be an input or output of the node that
    owns this spec. ``None`` leaves the tensor unspecified.
    """

    device: tuple[int, ...] = ()
    """The devices the tensor is distributed over, in shard order.

    The ``i``-th shard goes to ``device[i]``. Each entry is either a direct device
    index into :attr:`ModelConfiguration.device` (``>= 0``), or a key
    (conventionally negative) into :attr:`index_to_device_group_map`, which
    expands to a group of devices the shard is *replicated* across. An empty tuple
    leaves placement unspecified.
    """

    index_to_device_group_map: tuple[IndexToDeviceGroupMapEntry, ...] = ()
    """Optional expansion of the group keys used in :attr:`device`.

    Each :class:`IndexToDeviceGroupMapEntry` maps one key to the real device
    indices in that group. Only needed when a shard is replicated across a set of
    devices; direct device indices do not need an entry.
    """

    sharded_dim: tuple[ShardedDim, ...] = ()
    """One :class:`ShardedDim` per axis that is split.

    Axes not listed here are replicated across the devices in :attr:`device`. An
    empty tuple means the tensor is not split on any axis — i.e. it is fully
    replicated onto every device in :attr:`device` (or device group).
    """


@dataclasses.dataclass(frozen=True)
class NodeDeviceConfiguration:
    """The multi-device annotation of a single node under one configuration.

    Attached to :attr:`onnx_ir.Node.device_configurations`, it says how the node
    participates in a :class:`ModelConfiguration`: which of its tensors are
    sharded (tensor parallelism) and/or which pipeline stage it belongs to
    (pipeline parallelism). A node may carry several of these, one per
    configuration it takes part in.
    """

    configuration: ModelConfiguration | None = None
    """The :class:`ModelConfiguration` this annotation applies to.

    Bound by object identity (the proto ``configuration_id`` is derived from
    ``configuration.name`` on serialization). It should be one of the model's
    declared configurations. ``None`` leaves it unspecified.
    """

    sharding_spec: tuple[ShardingSpec, ...] = ()
    """The :class:`ShardingSpec` entries for this node's tensors.

    At most one per sharded input/output value. An empty tuple means the node is
    not tensor-sharded under this configuration (it may still be placed via
    :attr:`pipeline_stage`).
    """

    pipeline_stage: int | None = None
    """The pipeline stage this node belongs to, or ``None``.

    ``None`` means the node does not participate in pipeline parallelism under
    this configuration. Stages are non-negative integers; nodes sharing a stage
    run together, and a common convention maps ``stage`` to the device index in
    :attr:`ModelConfiguration.device`.
    """


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
            # ``num_devices`` is taken from the *registered* configuration object
            # so that device indices are validated against the model's ground
            # truth rather than a possibly-mismatched node-local reference.
            num_devices: int | None = None
            if config.configuration is None:
                errors.append(
                    f"Node '{node_label(node)}' has a device configuration without a "
                    "ModelConfiguration reference."
                )
            else:
                config_name = config.configuration.name
                registered = known_configs.get(config_name)
                if not config_name:
                    errors.append(
                        f"Node '{node_label(node)}' references a configuration with an "
                        "empty name (cannot be serialized)."
                    )
                    num_devices = config.configuration.num_devices
                elif registered is None:
                    errors.append(
                        f"Node '{node_label(node)}' references configuration "
                        f"'{config_name}' which is not declared in "
                        "model.device_configurations."
                    )
                    num_devices = config.configuration.num_devices
                elif config.configuration is not registered:
                    # Object-bound invariant: the node must reference the exact
                    # ModelConfiguration object registered on the model, not a
                    # same-named imposter (which would resolve to different
                    # ``num_devices`` after a serialization round-trip).
                    errors.append(
                        f"Node '{node_label(node)}' references a configuration object "
                        f"that is not the one registered under name '{config_name}' "
                        "on the model."
                    )
                    num_devices = registered.num_devices
                else:
                    num_devices = registered.num_devices
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
        # [0, num_devices), or a key into ``index_to_device_group_map`` that
        # names a group of real device ids — used to replicate a shard across a
        # set of devices. Validate accordingly.
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
