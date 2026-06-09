# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import onnx

import onnx_ir as ir
from onnx_ir import _multi_device


def _identity_model() -> tuple[ir.Model, ir.Node, ir.Value]:
    """Return a tiny ``x -> Relu -> y`` model along with its node and input."""
    x = ir.Value(name="x", shape=ir.Shape([4, 8]), type=ir.TensorType(ir.DataType.FLOAT))
    node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
    graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
    model = ir.Model(graph, ir_version=10)
    return model, node, x


class ModelConfigurationSerdeTest(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration"),
        "ModelProto.configuration is not available",
    )
    def test_model_configuration_serde_helpers_roundtrip(self):
        configuration = _multi_device.ModelConfiguration(
            name="conf0",
            num_devices=2,
            device=("CPU", "CUDA:0"),
        )

        proto = _multi_device.serialize_model_configuration(configuration)
        result = _multi_device.deserialize_model_configuration(proto)

        self.assertEqual(result, configuration)


class NodeDeviceConfigurationSerdeTest(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_derives_names_from_objects(self):
        x = ir.Value(name="x")
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration(name="conf0", num_devices=2),
            sharding_spec=(
                _multi_device.ShardingSpec(
                    value=x,
                    device=(0, 1),
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _multi_device.SimpleShardedDim(dim=4, num_shards=2),
                            ),
                        ),
                    ),
                ),
            ),
            pipeline_stage=1,
        )

        proto = _multi_device.serialize_node_device_configuration(config)

        self.assertEqual(proto.configuration_id, "conf0")
        self.assertEqual(proto.pipeline_stage, 1)
        self.assertEqual(proto.sharding_spec[0].tensor_name, "x")
        self.assertEqual(list(proto.sharding_spec[0].device), [0, 1])

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_deserialize_resolves_value_from_map(self):
        x = ir.Value(name="x")
        proto = onnx.NodeDeviceConfigurationProto()
        proto.configuration_id = "conf0"
        spec = proto.sharding_spec.add()
        spec.tensor_name = "x"

        result = _multi_device.deserialize_node_device_configuration(proto, values={"x": x})

        # The spec is bound to the actual value object.
        self.assertIs(result.sharding_spec[0].value, x)
        # The configuration is a placeholder carrying the id.
        self.assertEqual(result.configuration.name, "conf0")

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_deserialize_unresolved_value_creates_placeholder(self):
        proto = onnx.NodeDeviceConfigurationProto()
        spec = proto.sharding_spec.add()
        spec.tensor_name = "missing"

        result = _multi_device.deserialize_node_device_configuration(proto, values={})

        # A placeholder value carrying the name keeps the round-trip lossless.
        self.assertEqual(result.sharding_spec[0].value.name, "missing")
        reserialized = _multi_device.serialize_node_device_configuration(result)
        self.assertEqual(reserialized.sharding_spec[0].tensor_name, "missing")

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_simple_sharded_dim_with_symbolic_dim(self):
        x = ir.Value(name="x")
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration(name="conf0", num_devices=2),
            sharding_spec=(
                _multi_device.ShardingSpec(
                    value=x,
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _multi_device.SimpleShardedDim(
                                    dim=ir.SymbolicDim("BATCH"),
                                    num_shards=2,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        proto = _multi_device.serialize_node_device_configuration(config)
        self.assertEqual(
            proto.sharding_spec[0].sharded_dim[0].simple_sharding[0].dim_param, "BATCH"
        )

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_unnamed_value_raises(self):
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration("conf0", num_devices=1),
            sharding_spec=(_multi_device.ShardingSpec(value=ir.Value(name="")),),
        )
        with self.assertRaises(ValueError):
            _multi_device.serialize_node_device_configuration(config)


class ConvenienceApiTest(unittest.TestCase):
    def test_add_device_configuration_returns_and_registers(self):
        model, _, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        self.assertEqual(conf.num_devices, 2)
        self.assertEqual(model.device_configurations, (conf,))

    def test_add_device_configuration_duplicate_name_raises(self):
        model, _, _ = _identity_model()
        model.add_device_configuration("conf0", devices=("CPU",))
        with self.assertRaises(ValueError):
            model.add_device_configuration("conf0", devices=("CUDA:0",))

    def test_shard_records_and_infers_dim(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2, devices=(0, 1))

        specs = node.sharding_of(x)
        self.assertEqual(len(specs), 1)
        self.assertIs(specs[0].value, x)
        # dim is inferred from the value's static shape (4 at axis 0).
        self.assertEqual(specs[0].sharded_dim[0].simple_sharding[0].dim, 4)
        self.assertEqual(specs[0].sharded_dim[0].simple_sharding[0].num_shards, 2)

    def test_shard_groups_specs_under_same_configuration(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        y = node.outputs[0]
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        node.shard(y, configuration=conf, axis=0, num_shards=2)
        # Two different values -> two specs under a single NodeDeviceConfiguration.
        self.assertEqual(len(node.device_configurations), 1)
        self.assertEqual(len(node.device_configurations[0].sharding_spec), 2)

    def test_shard_same_value_multiple_axes_builds_one_spec(self):
        # Sharding the same value along two axes (a 2D device mesh) produces a
        # single ShardingSpec with one ShardedDim per axis.
        x = ir.Value(name="w", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        mesh = model.add_device_configuration("mesh2x2", num_devices=4)
        node.shard(x, configuration=mesh, axis=0, num_shards=2, devices=(0, 1, 2, 3))
        node.shard(x, configuration=mesh, axis=1, num_shards=2, devices=(0, 1, 2, 3))

        specs = node.sharding_of(x)
        self.assertEqual(len(specs), 1)
        self.assertEqual([d.axis for d in specs[0].sharded_dim], [0, 1])
        self.assertEqual(specs[0].device, (0, 1, 2, 3))

    def test_shard_same_value_unions_devices(self):
        x = ir.Value(name="w", shape=ir.Shape([8, 16]), type=ir.TensorType(ir.DataType.FLOAT))
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=11)
        mesh = model.add_device_configuration("mesh", num_devices=4)
        node.shard(x, configuration=mesh, axis=0, num_shards=2, devices=(0, 1))
        node.shard(x, configuration=mesh, axis=1, num_shards=2, devices=(1, 2, 3))
        # Devices are unioned, order-preserving, without duplicates.
        self.assertEqual(node.sharding_of(x)[0].device, (0, 1, 2, 3))

    def test_shard_same_axis_twice_raises(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        with self.assertRaises(ValueError):
            node.shard(x, configuration=conf, axis=0, num_shards=2)

    def test_shard_same_value_different_configurations(self):
        # Sharding the same value under two different configurations keeps the
        # specs in separate NodeDeviceConfigurations.
        model, node, x = _identity_model()
        a = model.add_device_configuration("a", devices=("CPU",))
        b = model.add_device_configuration("b", devices=("CUDA:0",))
        node.shard(x, configuration=a, axis=0, num_shards=2)
        node.shard(x, configuration=b, axis=1, num_shards=2)
        self.assertEqual(len(node.device_configurations), 2)
        self.assertEqual(len(node.sharding_of(x)), 2)

    def test_shard_rejects_foreign_value(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        foreign = ir.Value(name="z")
        with self.assertRaises(ValueError):
            node.shard(foreign, configuration=conf, axis=0, num_shards=2)

    def test_shard_accepts_negative_axis(self):
        # x has shape [4, 8]; axis=-1 means the last axis.
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.shard(x, configuration=conf, axis=-1, num_shards=2)
        spec = node.sharding_of(x)[0]
        self.assertEqual(spec.sharded_dim[0].axis, -1)
        # The dim is resolved from the last axis (size 8).
        self.assertEqual(spec.sharded_dim[0].simple_sharding[0].dim, 8)
        self.assertEqual(_multi_device.check_device_configurations(model), [])

    def test_shard_negative_axis_aliases_positive(self):
        # On a rank-2 tensor, axis=1 and axis=-1 are the same axis.
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.shard(x, configuration=conf, axis=1, num_shards=2)
        with self.assertRaises(ValueError):
            node.shard(x, configuration=conf, axis=-1, num_shards=2)

    def test_shard_rejects_out_of_range_negative_axis(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        # rank is 2, valid range is [-2, 1].
        with self.assertRaises(ValueError):
            node.shard(x, configuration=conf, axis=-3, num_shards=2)

    def test_shard_conflicting_pipeline_stage_raises(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        y = node.outputs[0]
        node.shard(x, configuration=conf, axis=0, num_shards=2, pipeline_stage=1)
        with self.assertRaises(ValueError):
            node.shard(y, configuration=conf, axis=0, num_shards=2, pipeline_stage=2)

    def test_shard_same_pipeline_stage_is_allowed(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        y = node.outputs[0]
        node.shard(x, configuration=conf, axis=0, num_shards=2, pipeline_stage=1)
        node.shard(y, configuration=conf, axis=0, num_shards=2, pipeline_stage=1)
        self.assertEqual(node.device_configurations[0].pipeline_stage, 1)

    def test_shard_rejects_bad_axis_and_shards(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        with self.assertRaises(ValueError):
            node.shard(x, configuration=conf, axis=5, num_shards=2)
        with self.assertRaises(ValueError):
            node.shard(x, configuration=conf, axis=0, num_shards=0)

    def test_sharding_follows_rename(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        x.name = "renamed"
        self.assertEqual(node.sharding_of(x)[0].value.name, "renamed")

    def test_shard_rejects_none_value(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        with self.assertRaises(ValueError):
            node.shard(None, configuration=conf, axis=0, num_shards=2)

    def test_shard_value_without_shape_infers_no_dim(self):
        # A value with unknown shape can still be sharded; dim is left unspecified.
        x = ir.Value(name="x")
        node = ir.Node("", "Relu", [x], outputs=[ir.Value(name="y")], name="relu0")
        graph = ir.Graph([x], [node.outputs[0]], nodes=[node], opset_imports={"": 18})
        model = ir.Model(graph, ir_version=10)
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.shard(x, configuration=conf, axis=3, num_shards=2)
        spec = node.sharding_of(x)[0]
        self.assertIsNone(spec.sharded_dim[0].simple_sharding[0].dim)

    def test_shard_updates_pipeline_stage_on_existing_configuration(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        node.shard(node.outputs[0], configuration=conf, axis=0, num_shards=2, pipeline_stage=3)
        self.assertEqual(node.device_configurations[0].pipeline_stage, 3)

    def test_add_device_configuration_explicit_num_devices(self):
        model, _, _ = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=4)
        self.assertEqual(conf.num_devices, 4)
        self.assertEqual(conf.device, ())

    def test_remove_device_configuration_by_object(self):
        model, _, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        removed = model.remove_device_configuration(conf)
        self.assertIs(removed, conf)
        self.assertEqual(model.device_configurations, ())

    def test_remove_device_configuration_by_name(self):
        model, _, _ = _identity_model()
        model.add_device_configuration("conf0", devices=("CPU",))
        other = model.add_device_configuration("conf1", devices=("CUDA:0",))
        model.remove_device_configuration("conf0")
        self.assertEqual(model.device_configurations, (other,))

    def test_remove_device_configuration_leaves_dangling_by_default(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        model.remove_device_configuration(conf)
        # The node reference is left intact and is reported by the checker.
        self.assertIs(node.device_configurations[0].configuration, conf)
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("conf0" in e for e in errors), errors)

    def test_remove_device_configuration_cascade_cleans_node_references(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        model.remove_device_configuration("conf0", cascade=True)
        self.assertEqual(node.device_configurations, ())
        self.assertEqual(_multi_device.check_device_configurations(model), [])

    def test_remove_device_configuration_cascade_keeps_other_configurations(self):
        model, node, x = _identity_model()
        keep = model.add_device_configuration("keep", devices=("CPU",))
        drop = model.add_device_configuration("drop", devices=("CUDA:0",))
        node.shard(x, configuration=keep, axis=0, num_shards=2)
        node.shard(node.outputs[0], configuration=drop, axis=0, num_shards=2)
        model.remove_device_configuration(drop, cascade=True)
        # Only the dropped configuration's node entry is removed.
        self.assertEqual(len(node.device_configurations), 1)
        self.assertIs(node.device_configurations[0].configuration, keep)

    def test_remove_device_configuration_cascade_covers_functions(self):
        model, _, _ = _identity_model()
        fx = ir.Value(name="fx")
        fnode = ir.Node("", "Relu", [fx], outputs=[ir.Value(name="fy")], name="frelu")
        fgraph = ir.Graph([fx], [fnode.outputs[0]], nodes=[fnode], opset_imports={"": 18})
        func = ir.Function(domain="custom", name="MyFunc", graph=fgraph, attributes=())
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        fnode.shard(fx, configuration=conf, axis=0, num_shards=2)
        model.functions[func.identifier()] = func
        model.remove_device_configuration(conf, cascade=True)
        self.assertEqual(fnode.device_configurations, ())

    def test_remove_unknown_configuration_raises(self):
        model, _, _ = _identity_model()
        with self.assertRaises(ValueError):
            model.remove_device_configuration("missing")
        stray = _multi_device.ModelConfiguration("stray", num_devices=1)
        with self.assertRaises(ValueError):
            model.remove_device_configuration(stray)


class CheckDeviceConfigurationsTest(unittest.TestCase):
    def test_valid_model_has_no_errors(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2, devices=(0, 1))
        self.assertEqual(_multi_device.check_device_configurations(model), [])

    def test_unknown_configuration_reported(self):
        model, node, x = _identity_model()
        # Configuration not registered on the model.
        stray = _multi_device.ModelConfiguration("ghost", num_devices=1)
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=stray,
                sharding_spec=(_multi_device.ShardingSpec(value=x),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("ghost" in e for e in errors), errors)

    def test_foreign_value_reported(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        foreign = ir.Value(name="foreign")
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(_multi_device.ShardingSpec(value=foreign),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("foreign" in e for e in errors), errors)

    def test_out_of_range_axis_and_device_reported(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(
                    _multi_device.ShardingSpec(
                        value=x,
                        device=(5,),  # only 2 devices
                        sharded_dim=(
                            _multi_device.ShardedDim(
                                axis=9,  # rank is 2
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=2),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("axis" in e for e in errors), errors)
        self.assertTrue(any("device index" in e for e in errors), errors)

    def test_negative_axis_within_range_is_valid(self):
        model, node, x = _identity_model()  # x rank 2
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(
                    _multi_device.ShardingSpec(
                        value=x,
                        sharded_dim=(
                            _multi_device.ShardedDim(
                                axis=-1,
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=2),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        self.assertEqual(_multi_device.check_device_configurations(model), [])

    def test_out_of_range_negative_axis_reported(self):
        model, node, x = _identity_model()  # x rank 2, valid [-2, 1]
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(
                    _multi_device.ShardingSpec(
                        value=x,
                        sharded_dim=(
                            _multi_device.ShardedDim(
                                axis=-3,
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=2),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("axis" in e for e in errors), errors)

    def test_duplicate_axis_within_spec_reported(self):
        model, node, x = _identity_model()  # x rank 2
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(
                    _multi_device.ShardingSpec(
                        value=x,
                        # axis 0 and axis -2 are the same axis on a rank-2 tensor.
                        sharded_dim=(
                            _multi_device.ShardedDim(
                                axis=0,
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=2),
                                ),
                            ),
                            _multi_device.ShardedDim(
                                axis=-2,
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=2),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("more than once" in e for e in errors), errors)


class CloneTest(unittest.TestCase):
    def test_clone_remaps_sharding_values(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU", "CUDA:0"))
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        node.shard(node.outputs[0], configuration=conf, axis=0, num_shards=2)

        cloned = model.clone()
        cloned_node = cloned.graph[0]
        specs = cloned_node.device_configurations[0].sharding_spec
        # Sharding points at the cloned graph's values, not the originals.
        self.assertIs(specs[0].value, cloned_node.inputs[0])
        self.assertIs(specs[1].value, cloned_node.outputs[0])
        self.assertIsNot(specs[0].value, x)
        self.assertEqual(_multi_device.check_device_configurations(cloned), [])

    def test_clone_node_without_configurations(self):
        # Cloning a node that has no device configurations is a no-op for them.
        model, _, _ = _identity_model()
        cloned = model.clone()
        self.assertEqual(cloned.graph[0].device_configurations, ())

    def test_clone_spec_with_unmapped_value_is_preserved(self):
        # A ShardingSpec whose value is None (not in the value map) is preserved
        # as-is, and a configuration with no remapped spec is left unchanged.
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        original = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(_multi_device.ShardingSpec(value=None, device=(0,)),),
            ),
        )
        node.device_configurations = original
        cloned = model.clone()
        cloned_config = cloned.graph[0].device_configurations[0]
        self.assertIsNone(cloned_config.sharding_spec[0].value)
        self.assertEqual(cloned_config.sharding_spec[0].device, (0,))

    def test_clone_propagates_dropped_value_to_none(self):
        # When the cloner is told a value maps to None (intentionally dropped),
        # the sharding reference follows to None rather than staying stale.
        from onnx_ir import _cloner

        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", num_devices=2)
        node.shard(x, configuration=conf, axis=0, num_shards=2)

        cloner = _cloner.Cloner(
            attr_map={},
            value_map={x: None},
            metadata_props={},
        )
        cloned_configs = cloner._remap_device_configurations(node.device_configurations)
        self.assertIsNone(cloned_configs[0].sharding_spec[0].value)


class SerdeHelperEdgeCaseTest(unittest.TestCase):
    """Cover the lower-level serde helper branches directly."""

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_simple_sharded_dim_unspecified(self):
        # dim is None -> neither dim_value nor dim_param is set.
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration("conf0", num_devices=1),
            sharding_spec=(
                _multi_device.ShardingSpec(
                    value=ir.Value(name="x"),
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _multi_device.SimpleShardedDim(dim=None, num_shards=2),
                            ),
                        ),
                    ),
                ),
            ),
        )
        proto = _multi_device.serialize_node_device_configuration(config)
        simple = proto.sharding_spec[0].sharded_dim[0].simple_sharding[0]
        self.assertFalse(simple.HasField("dim_value"))
        self.assertFalse(simple.HasField("dim_param"))
        self.assertEqual(simple.num_shards, 2)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_skips_symbolic_dim_without_value(self):
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration("conf0", num_devices=1),
            sharding_spec=(
                _multi_device.ShardingSpec(
                    value=ir.Value(name="x"),
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _multi_device.SimpleShardedDim(
                                    dim=ir.SymbolicDim(None), num_shards=1
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        proto = _multi_device.serialize_node_device_configuration(config)
        simple = proto.sharding_spec[0].sharded_dim[0].simple_sharding[0]
        self.assertFalse(simple.HasField("dim_param"))

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_deserialize_dim_param_and_unspecified(self):
        proto = onnx.NodeDeviceConfigurationProto()
        spec = proto.sharding_spec.add()
        spec.tensor_name = "x"
        sharded = spec.sharded_dim.add()
        sharded.axis = 0
        sharded.simple_sharding.add(dim_param="BATCH", num_shards=2)
        sharded.simple_sharding.add(num_shards=1)  # neither dim_value nor dim_param

        result = _multi_device.deserialize_node_device_configuration(proto)
        simple = result.sharding_spec[0].sharded_dim[0].simple_sharding
        self.assertEqual(simple[0].dim, ir.SymbolicDim("BATCH"))
        self.assertIsNone(simple[1].dim)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_deserialize_empty_tensor_name_resolves_to_none(self):
        proto = onnx.NodeDeviceConfigurationProto()
        proto.sharding_spec.add()  # tensor_name left empty
        result = _multi_device.deserialize_node_device_configuration(proto, values={})
        self.assertIsNone(result.sharding_spec[0].value)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_deserialize_without_values_argument(self):
        proto = onnx.NodeDeviceConfigurationProto()
        spec = proto.sharding_spec.add()
        spec.tensor_name = "x"
        # values defaults to None -> treated as empty mapping.
        result = _multi_device.deserialize_node_device_configuration(proto)
        self.assertEqual(result.sharding_spec[0].value.name, "x")

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_spec_without_value(self):
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration("conf0", num_devices=1),
            sharding_spec=(_multi_device.ShardingSpec(value=None, device=(0,)),),
        )
        proto = _multi_device.serialize_node_device_configuration(config)
        self.assertEqual(proto.sharding_spec[0].tensor_name, "")
        self.assertEqual(list(proto.sharding_spec[0].device), [0])

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_serialize_config_with_empty_name_raises(self):
        config = _multi_device.NodeDeviceConfiguration(
            configuration=_multi_device.ModelConfiguration("", num_devices=1),
        )
        with self.assertRaises(ValueError):
            _multi_device.serialize_node_device_configuration(config)


class CheckEdgeCaseTest(unittest.TestCase):
    """Cover the remaining branches of check_device_configurations."""

    def test_node_without_configurations_is_skipped(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        # Add a second, unsharded node.
        extra = ir.Node("", "Identity", [node.outputs[0]], num_outputs=1, name="id1")
        model.graph.append(extra)
        node.shard(x, configuration=conf, axis=0, num_shards=2)
        self.assertEqual(_multi_device.check_device_configurations(model), [])

    def test_function_node_is_checked(self):
        model, _, _ = _identity_model()
        # Build a function with a sharded node that references an unknown config.
        fx = ir.Value(name="fx")
        fnode = ir.Node("", "Relu", [fx], outputs=[ir.Value(name="fy")], name="frelu")
        fgraph = ir.Graph([fx], [fnode.outputs[0]], nodes=[fnode], opset_imports={"": 18})
        func = ir.Function(domain="custom", name="MyFunc", graph=fgraph, attributes=())
        fnode.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=_multi_device.ModelConfiguration("ghost", num_devices=1),
                sharding_spec=(_multi_device.ShardingSpec(value=fx),),
            ),
        )
        model.functions[func.identifier()] = func
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("ghost" in e for e in errors), errors)

    def test_serialize_rejects_non_dataclass_node_configuration(self):
        # The strict type contract: only NodeDeviceConfiguration is accepted;
        # raw bytes or protos are rejected at the serialization boundary.
        import onnx_ir as ir_module
        from onnx_ir import serde

        model, node, _ = _identity_model()
        model.add_device_configuration("conf0", devices=("CPU",))
        node.device_configurations = (b"\x08\x01",)  # not a NodeDeviceConfiguration
        with self.assertRaises((TypeError, ir_module.serde.SerdeError)):
            serde.serialize_model(model)

    def test_serialize_rejects_non_dataclass_model_configuration(self):
        from onnx_ir import serde

        model, _, _ = _identity_model()
        model.device_configurations = (b"\x08\x01",)  # not a ModelConfiguration
        with self.assertRaises((TypeError, serde.SerdeError)):
            serde.serialize_model(model)

    def test_missing_configuration_reference_reported(self):
        model, node, x = _identity_model()
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=None,
                sharding_spec=(_multi_device.ShardingSpec(value=x),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("without a" in e for e in errors), errors)

    def test_empty_configuration_name_reported(self):
        model, node, x = _identity_model()
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=_multi_device.ModelConfiguration("", num_devices=1),
                sharding_spec=(_multi_device.ShardingSpec(value=x),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("empty name" in e for e in errors), errors)

    def test_spec_without_value_reported(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(_multi_device.ShardingSpec(value=None),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("without a value" in e for e in errors), errors)

    def test_spec_value_with_empty_name_reported(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        unnamed = ir.Value(name="")
        node._inputs = (*node.inputs, unnamed)  # make it part of node IO
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(_multi_device.ShardingSpec(value=unnamed),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("empty name" in e for e in errors), errors)

    def test_num_shards_below_one_reported(self):
        model, node, x = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=conf,
                sharding_spec=(
                    _multi_device.ShardingSpec(
                        value=x,
                        sharded_dim=(
                            _multi_device.ShardedDim(
                                axis=0,
                                simple_sharding=(
                                    _multi_device.SimpleShardedDim(num_shards=0),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertTrue(any("num_shards" in e for e in errors), errors)

    def test_device_index_not_checked_without_configuration(self):
        # configuration is None -> num_devices is None -> device range check is skipped,
        # but the missing-configuration error is still reported.
        model, node, x = _identity_model()
        node.device_configurations = (
            _multi_device.NodeDeviceConfiguration(
                configuration=None,
                sharding_spec=(_multi_device.ShardingSpec(value=x, device=(99,)),),
            ),
        )
        errors = _multi_device.check_device_configurations(model)
        self.assertFalse(any("device index" in e for e in errors), errors)
        self.assertTrue(any("without a" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
