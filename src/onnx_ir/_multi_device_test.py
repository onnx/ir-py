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
        # Both specs live under a single NodeDeviceConfiguration.
        self.assertEqual(len(node.device_configurations), 1)
        self.assertEqual(len(node.device_configurations[0].sharding_spec), 2)

    def test_shard_rejects_foreign_value(self):
        model, node, _ = _identity_model()
        conf = model.add_device_configuration("conf0", devices=("CPU",))
        foreign = ir.Value(name="z")
        with self.assertRaises(ValueError):
            node.shard(foreign, configuration=conf, axis=0, num_shards=2)

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


if __name__ == "__main__":
    unittest.main()
