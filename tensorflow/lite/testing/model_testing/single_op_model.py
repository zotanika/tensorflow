import argparse
import sys
import os
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from typing import Type
import subprocess


@tf.function
def add(a, b):
    return tf.add(a, b)


@tf.function
def mul(a, b):
    return tf.multiply(a, b)


@tf.function
def concat(a, b):
    return tf.concat([a, b], 0)


@tf.function
def transpose(a):
    return tf.transpose(a, conjugate=False)


@tf.function
def strided_slice(a):
    return tf.strided_slice(a, [1, 0, 0], [2, 2, 3], [1, 1, 1])


fully_connected_layer = tf.keras.layers.Dense(32, activation="relu")


@tf.function
def fully_connected(a):
    return fully_connected_layer(a)


conv2d_layer = tf.keras.layers.Conv2D(512, 3)
@tf.function
def conv2d(a):
    return conv2d_layer(a)


@tf.function
def slice(a):
    return tf.slice(a, [1,0,0], [1,1,3])


depthwise_conv2d_layer = tf.keras.layers.DepthwiseConv2D(8)
@tf.function
def depthwise_conv2d(a):
    return depthwise_conv2d_layer(a)


@tf.function
def maxpool2d(a):
    return tf.keras.layers.MaxPool2D(pool_size=(2,2))(a)


@tf.function
def pad(a):
    return tf.keras.layers.ZeroPadding2D(padding=1)(a)


@tf.function
def strided_slice(a):
    return tf.strided_slice(a, [1,0,0], [2,2,3], [1,1,1])


@tf.function
def gather(a):
    indices = [2, 0, 2, 5]
    return tf.gather(a, indices)

@tf.function
def pack(a, b):
    return tf.stack([a,b])

@tf.function
def unpack(a):
    return tf.unstack(a)

@tf.function
def cast(a):
    return tf.cast(a, tf.uint32)

@dataclass
class Tensor:
    name: str
    shape: tf.TensorShape
    dtype: tf.DType


@dataclass
class Ops:
    func: tf.function
    input_tensors: Type[Tensor]


OPS = {
    "add": Ops(add, [Tensor("a", [2, 2], tf.float32), Tensor("b", [2, 2], tf.float32)]),
    "mul": Ops(mul, [Tensor("a", [2, 2], tf.float32), Tensor("b", [2, 2], tf.float32)]),
    "concat": Ops(
        concat, [Tensor("a", [2, 2], tf.float32), Tensor("b", [2, 2], tf.float32)]
    ),
    "transpose": Ops(transpose, [Tensor("a", [3, 2], tf.float32)]),
    "strided_slice": Ops(strided_slice, [Tensor("a", [3, 3, 3], tf.float32)]),
    "fully_connected": Ops(fully_connected, [Tensor("a", [3, 3, 3], tf.float32)]),
    "conv2d": Ops(conv2d, [Tensor("a", [1, 32, 32, 3], tf.float32)]),
    "slice" : Ops(slice, [Tensor("a", [3,3,3], tf.float32)]),
    "depthwise_conv2d" : Ops(depthwise_conv2d, [Tensor("a", [1, 32, 32, 32], tf.float32)]),
    "maxpool2d" : Ops(maxpool2d, [Tensor("a", [1, 32, 32, 1], tf.float32)]),
    "pad" : Ops(pad, [Tensor("a", [1, 3, 3, 1], tf.float32)]),
    "strided_slice" : Ops(strided_slice, [Tensor("a", [3, 2, 3], tf.float32)]),
    "gather" : Ops(gather, [Tensor("a", [6], tf.float32)]),
    "pack" : Ops(pack, [Tensor("a", [1,2], tf.float32), Tensor("b", [1,2], tf.float32)]),
    "unpack" : Ops(unpack, [Tensor("a", [3,4], tf.float32)]),
    "cast" : Ops(cast, [Tensor("a", [2,3,4], tf.float32)]),
}


def __generate_fp16_quant_model(op_name, tf_func, input_tensors, outdir):
    tspecs = []
    for tensor in input_tensors:
        tspecs.append(tf.TensorSpec(tensor.shape, tensor.dtype, name=tensor.name))

    concrete_func = tf_func.get_concrete_function(*tspecs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    fpath = os.path.join(outdir, f"{op_name}_fp16_quant.tflite")
    with open(fpath, "wb") as f:
        f.write(tflite_model)
    return fpath


def __generate_int8_quant_model(op_name, tf_func, input_tensors, outdir):
    tspecs = []
    for tensor in input_tensors:
        tspecs.append(tf.TensorSpec(tensor.shape, tensor.dtype, name=tensor.name))

    concrete_func = tf_func.get_concrete_function(*tspecs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    def representative_datasetV2(input_tensors):
        calibration_inputs = {}
        for tensor in input_tensors:
            if tensor.shape:
                dims = tensor.shape
                calibration_inputs[tensor.name] = np.random.uniform(
                    low=0, high=1, size=dims
                ).astype(tensor.dtype.as_numpy_dtype)
        return calibration_inputs

    def representative_datasetV1(input_tensors):
        calibration_inputs = []
        for tensor in input_tensors:
            if tensor.shape:
                dims = tensor.shape
                calibration_inputs.append(
                    np.random.uniform(low=0, high=1, size=dims).astype(
                        tensor.dtype.as_numpy_dtype
                    )
                )
        return calibration_inputs

    def representative_dataset_gen():
        for _ in range(100):
            yield representative_datasetV1(input_tensors)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    fpath = os.path.join(outdir, f"{op_name}_int8_quant.tflite")
    with open(fpath, "wb") as f:
        f.write(tflite_model)

    return fpath


def __generate_model(op_name, tf_func, input_tensors, outdir, dtype=None):
    tspecs = []
    dtype = input_tensors[0].dtype if dtype == None else dtype
    for tensor in input_tensors:
        tspecs.append(tf.TensorSpec(tensor.shape, dtype, name=tensor.name))

    concrete_func = tf_func.get_concrete_function(*tspecs)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    tflite_model = converter.convert()
    suffix = {tf.float32:"", tf.float16:"_fp16", tf.int8:"_int8"}[dtype]
    fpath = os.path.join(outdir, f"{op_name}{suffix}.tflite")
    with open(fpath, "wb") as f:
        f.write(tflite_model)
    return fpath


def tflite_convert_from_concrete_function(op_name, tf_func, input_tensors, outdir):
    generated_files = []

    # original model
    generated_files.append(__generate_model(op_name, tf_func, input_tensors, outdir))

    # fp16 quant model
    generated_files.append(__generate_fp16_quant_model(op_name, tf_func, input_tensors, outdir))

    # int8 quant model
    generated_files.append(__generate_int8_quant_model(op_name, tf_func, input_tensors, outdir))

    return generated_files


def generate_qnn_code(op, tflite_files):
    sdk_root = os.environ["QNN_SDK_ROOT"]
    if not sdk_root:
        print("QNN_SDK_ROOT not set!")
        return

    result = []
    for entry in tflite_files:
        input_args = []
        for tensor in op.input_tensors:
            input_args.append("--input_dim")
            input_args.append(tensor.name)
            input_args.append(",".join(map(str, tensor.shape)))
        outfile = entry + ".cpp"
        cmd = [
            f"{sdk_root}/target/x86_64-linux-clang/bin/qnn-tflite-converter",
            "--input_network",
            entry,
            *input_args,
            "--out_node",
            "Identity",
            "--output_path",
            outfile,
        ]
        print(cmd)
        ret = subprocess.run(cmd, stdout=subprocess.PIPE)
        if ret.returncode == 0:
            result.append(outfile)

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("OPNAME", action="store", help="Name of the op")
    parser.add_argument(
        "--outdir", action="store", default=None, help="Output directory"
    )
    parser.add_argument(
        "--qnn", action="store_true", default=False, help="Generate QNN code"
    )

    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        exit(0)

    if args.outdir is None:
        args.outdir = os.path.dirname(__file__)

    return args


def main(args):
    op_name = args.OPNAME
    tflite_files = tflite_convert_from_concrete_function(
        op_name, OPS[op_name].func, OPS[op_name].input_tensors, args.outdir
    )
    qnn_files = None
    if args.qnn:
        qnn_files = generate_qnn_code(OPS[op_name], tflite_files)


    print("\nGENERATED FILES : ")
    if tflite_files:
        print(f"\tTFLite files : {tflite_files}")
    if qnn_files:
        print(f"\tQNN files : {qnn_files}")


if __name__ == "__main__":
    main(parse_args())
