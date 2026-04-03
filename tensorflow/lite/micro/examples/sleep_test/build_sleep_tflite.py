#!/usr/bin/env python3
"""
Build a TFLM-compatible .tflite for the sleep monitoring model.

The original model uses Bidirectional GRU (Keras), which TFLite converter
turns into WHILE + SLICE + CONCAT ops to emulate TensorArray. This pattern
has O(n²) complexity and crashes on TFLM due to dynamic shape issues.

This script rewrites the original .tflite by:
  1. Extracting GRU weights from the WHILE body subgraphs
  2. Replacing the WHILE-based BiGRU blocks with a single custom op
     "BIDIRECTIONAL_SEQUENCE_GRU" per BiGRU layer
  3. Keeping all other ops (LayerNorm, Dense, ELU, Softmax) unchanged
  4. Removing the WHILE cond/body subgraphs (1-8)

The resulting model runs on TFLM with the custom BiGRU kernel from:
  tensorflow/lite/micro/kernels/bidirectional_sequence_gru.cc

Usage:
  python3 build_sleep_tflite.py <original.tflite> <output.tflite>

Example:
  python3 build_sleep_tflite.py \\
    ~/work/models/sleeping/funt_apple_2026-01-01-12-37-22.tflite \\
    ~/work/models/sleeping/sleep_full.tflite
"""

import argparse
import sys

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils


# BiGRU weight layout inside WHILE body subgraphs.
# Each WHILE body has:
#   T#9  = recurrent_kernel [3*hidden, hidden]
#   T#7  = recurrent_bias   [3*hidden]
#   T#10 = input_kernel     [3*hidden, input_dim]
#   T#6  = input_bias       [3*hidden]
BODY_TENSOR_RECURRENT_KERNEL = 9
BODY_TENSOR_RECURRENT_BIAS = 7
BODY_TENSOR_INPUT_KERNEL = 10
BODY_TENSOR_INPUT_BIAS = 6

# Original model op ranges:
#   0-14:  LayerNorm1 + Dense1 + ELU
#   15-22: BiGRU1 (RESHAPE + REVERSE + 2xWHILE + RESHAPE + CONCAT)
#   23-30: BiGRU2 (same pattern)
#   31-47: LayerNorm2 + Dense2 + ELU + Dense3 + Softmax
BIGRU1_OP_START = 15
BIGRU1_OP_END = 23  # exclusive
BIGRU2_OP_START = 23
BIGRU2_OP_END = 31

# BiGRU1: input tensor = 47 (ELU output), output tensor = 63 (CONCAT output)
# BiGRU2: input tensor = 63, output tensor = 79
BIGRU1_INPUT_TENSOR = 47
BIGRU1_OUTPUT_TENSOR = 63
BIGRU2_INPUT_TENSOR = 63
BIGRU2_OUTPUT_TENSOR = 79

# Subgraph mapping: sg2=BiGRU1 backward body, sg4=BiGRU1 forward body,
#                    sg6=BiGRU2 backward body, sg8=BiGRU2 forward body
BIGRU1_FW_BODY_SG = 4
BIGRU1_BW_BODY_SG = 2
BIGRU2_FW_BODY_SG = 8
BIGRU2_BW_BODY_SG = 6


def extract_weight(model, sg_idx, tensor_idx):
    """Extract weight data from a subgraph tensor."""
    tensor = model.subgraphs[sg_idx].tensors[tensor_idx]
    buf = model.buffers[tensor.buffer]
    return np.frombuffer(bytes(buf.data), dtype=np.float32).reshape(tensor.shape)


def add_weight_tensor(model, sg, name, data):
    """Add a float32 weight tensor to the model, return its tensor index."""
    buf = schema.BufferT()
    buf.data = list(data.astype(np.float32).tobytes())
    model.buffers.append(buf)

    t = schema.TensorT()
    t.name = name.encode()
    t.shape = list(data.shape)
    t.type = 0  # FLOAT32
    t.buffer = len(model.buffers) - 1
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def make_bigru_op(opcode_idx, input_tensor, output_tensor, weight_indices):
    """Create a BIDIRECTIONAL_SEQUENCE_GRU operator."""
    op = schema.OperatorT()
    op.opcodeIndex = opcode_idx
    op.inputs = [input_tensor] + weight_indices
    op.outputs = [output_tensor]
    return op


def build(orig_path, out_path):
    model = flatbuffer_utils.read_model(orig_path)
    sg = model.subgraphs[0]
    orig_op_count = len(sg.operators)

    # Add custom opcode
    oc = schema.OperatorCodeT()
    oc.deprecatedBuiltinCode = 32  # CUSTOM
    oc.builtinCode = 32
    oc.customCode = b"BIDIRECTIONAL_SEQUENCE_GRU"
    oc.version = 1
    model.operatorCodes.append(oc)
    bigru_opcode = len(model.operatorCodes) - 1

    # Extract GRU weights from WHILE body subgraphs and add as new tensors
    bigru_configs = [
        (
            "bigru1",
            BIGRU1_FW_BODY_SG,
            BIGRU1_BW_BODY_SG,
            BIGRU1_INPUT_TENSOR,
            BIGRU1_OUTPUT_TENSOR,
        ),
        (
            "bigru2",
            BIGRU2_FW_BODY_SG,
            BIGRU2_BW_BODY_SG,
            BIGRU2_INPUT_TENSOR,
            BIGRU2_OUTPUT_TENSOR,
        ),
    ]

    new_ops = list(sg.operators[:BIGRU1_OP_START])

    for name, fw_sg, bw_sg, in_t, out_t in bigru_configs:
        fw_ik = extract_weight(model, fw_sg, BODY_TENSOR_INPUT_KERNEL)
        fw_ib = extract_weight(model, fw_sg, BODY_TENSOR_INPUT_BIAS)
        fw_rk = extract_weight(model, fw_sg, BODY_TENSOR_RECURRENT_KERNEL)
        fw_rb = extract_weight(model, fw_sg, BODY_TENSOR_RECURRENT_BIAS)
        bw_ik = extract_weight(model, bw_sg, BODY_TENSOR_INPUT_KERNEL)
        bw_ib = extract_weight(model, bw_sg, BODY_TENSOR_INPUT_BIAS)
        bw_rk = extract_weight(model, bw_sg, BODY_TENSOR_RECURRENT_KERNEL)
        bw_rb = extract_weight(model, bw_sg, BODY_TENSOR_RECURRENT_BIAS)

        weight_indices = [
            add_weight_tensor(model, sg, f"{name}_fw_ik", fw_ik),
            add_weight_tensor(model, sg, f"{name}_fw_ib", fw_ib),
            add_weight_tensor(model, sg, f"{name}_fw_rk", fw_rk),
            add_weight_tensor(model, sg, f"{name}_fw_rb", fw_rb),
            add_weight_tensor(model, sg, f"{name}_bw_ik", bw_ik),
            add_weight_tensor(model, sg, f"{name}_bw_ib", bw_ib),
            add_weight_tensor(model, sg, f"{name}_bw_rk", bw_rk),
            add_weight_tensor(model, sg, f"{name}_bw_rb", bw_rb),
        ]

        new_ops.append(make_bigru_op(bigru_opcode, in_t, out_t, weight_indices))

    new_ops.extend(sg.operators[BIGRU2_OP_END:])
    sg.operators = new_ops

    model.subgraphs = [sg]

    flatbuffer_utils.write_model(model, out_path)

    print(f"Input:  {orig_path}")
    print(f"Output: {out_path}")
    print(f"Ops:    {orig_op_count} -> {len(sg.operators)}")
    print(
        f"  Removed: {BIGRU1_OP_END - BIGRU1_OP_START} + {BIGRU2_OP_END - BIGRU2_OP_START} WHILE-based ops"
    )
    print(f"  Added:   2 BIDIRECTIONAL_SEQUENCE_GRU custom ops")
    print(f"Subgraphs: {len(model.subgraphs)} (removed {8} WHILE cond/body subgraphs)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert sleep model from WHILE-based BiGRU to custom op BiGRU"
    )
    parser.add_argument("input", help="Original .tflite with WHILE-based BiGRU")
    parser.add_argument("output", help="Output .tflite with custom BiGRU op")
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
