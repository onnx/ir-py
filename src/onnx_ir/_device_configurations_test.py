# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import onnx

import onnx_ir as ir
from onnx_ir import _device_configurations


class DeviceConfigurationsTest(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(onnx.ModelProto(), "configuration"),
        "ModelProto.configuration is not available",
    )
    def test_model_configuration_serde_helpers_roundtrip(self):
        configuration = _device_configurations.ModelConfiguration(
            name="conf0",
            num_devices=2,
            device=("CPU", "CUDA:0"),
        )

        proto = _device_configurations.serialize_model_configuration(configuration)
        result = _device_configurations.deserialize_model_configuration(proto)

        self.assertEqual(result, configuration)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_node_device_configuration_serde_helpers_roundtrip(self):
        node_device_configuration = _device_configurations.NodeDeviceConfiguration(
            configuration_id="conf0",
            sharding_spec=(
                _device_configurations.ShardingSpec(
                    tensor_name="x",
                    device=(0, 1),
                    index_to_device_group_map=(
                        _device_configurations.IndexToDeviceGroupMapEntry(
                            key=0,
                            value=(1, 0),
                        ),
                    ),
                    sharded_dim=(
                        _device_configurations.ShardedDim(
                            axis=0,
                            simple_sharding=(
                                _device_configurations.SimpleShardedDim(
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

        proto = _device_configurations.serialize_node_device_configuration(
            node_device_configuration
        )
        result = _device_configurations.deserialize_node_device_configuration(proto)

        self.assertEqual(result, node_device_configuration)

    @unittest.skipUnless(
        hasattr(onnx.NodeProto(), "device_configurations"),
        "NodeProto.device_configurations is not available",
    )
    def test_simple_sharded_dim_with_symbolic_dim(self):
        """SimpleShardedDim should round-trip a SymbolicDim."""
        simple_dim = _device_configurations.SimpleShardedDim(
            dim=ir.SymbolicDim("BATCH"),
            num_shards=2,
        )
        node_device_configuration = _device_configurations.NodeDeviceConfiguration(
            configuration_id="conf0",
            sharding_spec=(
                _device_configurations.ShardingSpec(
                    tensor_name="x",
                    sharded_dim=(
                        _device_configurations.ShardedDim(
                            axis=0,
                            simple_sharding=(simple_dim,),
                        ),
                    ),
                ),
            ),
        )

        proto = _device_configurations.serialize_node_device_configuration(
            node_device_configuration
        )
        result = _device_configurations.deserialize_node_device_configuration(proto)

        self.assertEqual(result, node_device_configuration)
