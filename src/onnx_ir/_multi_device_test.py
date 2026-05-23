# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import onnx

import onnx_ir as ir
from onnx_ir import _multi_device


class DeviceConfigurationsTest(unittest.TestCase):
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

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_node_device_configuration_serde_helpers_roundtrip(self):
        node_device_configuration = _multi_device.NodeDeviceConfiguration(
            configuration_id="conf0",
            sharding_spec=(
                _multi_device.ShardingSpec(
                    tensor_name="x",
                    device=(0, 1),
                    index_to_device_group_map=(
                        _multi_device.IndexToDeviceGroupMapEntry(
                            key=0,
                            value=(1, 0),
                        ),
                    ),
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _multi_device.SimpleShardedDim(
                                    dim=4,
                                    num_shards=2,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            pipeline_stage=1,
        )

        proto = _multi_device.serialize_node_device_configuration(node_device_configuration)
        result = _multi_device.deserialize_node_device_configuration(proto)

        self.assertEqual(result, node_device_configuration)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_simple_sharded_dim_with_symbolic_dim(self):
        """SimpleShardedDim should round-trip a SymbolicDim."""
        simple_dim = _multi_device.SimpleShardedDim(
            dim=ir.SymbolicDim("BATCH"),
            num_shards=2,
        )
        node_device_configuration = _multi_device.NodeDeviceConfiguration(
            configuration_id="conf0",
            sharding_spec=(
                _multi_device.ShardingSpec(
                    tensor_name="x",
                    sharded_dim=(
                        _multi_device.ShardedDim(
                            axis=0,
                            simple_sharding=(simple_dim,),
                        ),
                    ),
                ),
            ),
        )

        proto = _multi_device.serialize_node_device_configuration(node_device_configuration)
        result = _multi_device.deserialize_node_device_configuration(proto)

        self.assertEqual(result, node_device_configuration)
