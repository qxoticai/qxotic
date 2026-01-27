package ai.qxotic.jota.hip;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.tensor.ExpressionGraph;
import java.util.Objects;

final class HipReductionKernelGenerator {

    KernelProgramSpec generate(ExpressionGraph.ReductionInfo reduction, Shape inputShape) {
        Objects.requireNonNull(reduction, "reduction");
        Objects.requireNonNull(inputShape, "inputShape");

        DataType dataType = reduction.dataType();
        if (dataType != DataType.FP32 && dataType != DataType.FP64) {
            throw new UnsupportedOperationException(
                    "HIP reduction supports FP32/FP64 only: " + dataType);
        }
        if (reduction.op() != ai.qxotic.jota.tensor.ReductionOp.SUM
                && reduction.op() != ai.qxotic.jota.tensor.ReductionOp.MIN
                && reduction.op() != ai.qxotic.jota.tensor.ReductionOp.MAX) {
            throw new UnsupportedOperationException(
                    "Unsupported HIP reduction op: " + reduction.op().name());
        }

        long[] inDims = inputShape.toArray();
        int rank = inDims.length;
        int axis = reduction.axis();
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Invalid reduction axis: " + axis);
        }

        long[] outDims = outputDims(inDims, axis, reduction.keepDims());
        long[] inStride = rowMajorStride(inDims);
        long[] outStride = rowMajorStride(outDims);
        long reduceDim = inDims[axis];
        long outSize = size(outDims);

        String kernelName = kernelName(reduction, inputShape);
        String typeName = dataType == DataType.FP64 ? "double" : "float";
        String initValue = dataType == DataType.FP64 ? "0.0" : "0.0f";
        String minFn = dataType == DataType.FP64 ? "fmin" : "fminf";
        String maxFn = dataType == DataType.FP64 ? "fmax" : "fmaxf";

        StringBuilder source = new StringBuilder();
        source.append("#include <hip/hip_runtime.h>\n");
        source.append("#include <stdint.h>\n");
        source.append("extern \"C\" __global__ void ")
                .append(kernelName)
                .append("(const ")
                .append(typeName)
                .append(" *input, ")
                .append(typeName)
                .append(" *output) {\n");
        source.append("  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
        source.append("  if (idx >= ").append(outSize).append("LL) return;\n");
        source.append("  long long tmp = idx;\n");

        for (int i = 0; i < outDims.length; i++) {
            long stride = outStride[i];
            source.append("  long long o").append(i).append(" = tmp / ")
                    .append(stride).append("LL;\n");
            if (i < outDims.length - 1) {
                source.append("  tmp = tmp % ").append(stride).append("LL;\n");
            }
        }

        source.append("  long long base = 0;\n");
        for (int i = 0; i < rank; i++) {
            if (i == axis) {
                continue;
            }
            int outIndex = reduction.keepDims() ? i : (i < axis ? i : i - 1);
            source.append("  base += o")
                    .append(outIndex)
                    .append(" * ")
                    .append(inStride[i])
                    .append("LL;\n");
        }

        source.append("  if (").append(reduceDim).append("LL == 0) {\n");
        source.append("    output[idx] = ").append(initValue).append(";\n");
        source.append("    return;\n");
        source.append("  }\n");
        if (reduction.op() == ai.qxotic.jota.tensor.ReductionOp.SUM) {
            source.append("  ").append(typeName).append(" acc = ").append(initValue).append(";\n");
            source.append("  for (int r = 0; r < ").append(reduceDim).append("; r++) {\n");
            source.append("    acc += input[base + (long long)r * ")
                    .append(inStride[axis])
                    .append("LL];\n");
            source.append("  }\n");
        } else {
            source.append("  ").append(typeName).append(" acc = input[base];\n");
            source.append("  for (int r = 1; r < ").append(reduceDim).append("; r++) {\n");
            source.append("    ").append(typeName).append(" v = input[base + (long long)r * ")
                    .append(inStride[axis])
                    .append("LL];\n");
            if (reduction.op() == ai.qxotic.jota.tensor.ReductionOp.MIN) {
                source.append("    acc = ").append(minFn).append("(acc, v);\n");
            } else {
                source.append("    acc = ").append(maxFn).append("(acc, v);\n");
            }
            source.append("  }\n");
        }
        source.append("  output[idx] = acc;\n");
        source.append("}\n");

        return new KernelProgramSpec(kernelName, source.toString());
    }

    private static long[] outputDims(long[] inputDims, int axis, boolean keepDims) {
        if (keepDims) {
            long[] out = inputDims.clone();
            out[axis] = 1;
            return out;
        }
        long[] out = new long[inputDims.length - 1];
        for (int i = 0, j = 0; i < inputDims.length; i++) {
            if (i == axis) {
                continue;
            }
            out[j++] = inputDims[i];
        }
        return out;
    }

    private static long[] rowMajorStride(long[] dims) {
        long[] stride = new long[dims.length];
        long acc = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            stride[i] = acc;
            acc *= dims[i];
        }
        return stride;
    }

    private static long size(long[] dims) {
        long prod = 1;
        for (long dim : dims) {
            prod *= dim;
        }
        return prod;
    }

    private static String kernelName(ExpressionGraph.ReductionInfo reduction, Shape inputShape) {
        int hash = Objects.hash(reduction.op(), reduction.axis(), reduction.keepDims(), inputShape);
        String suffix = reduction.dataType() == DataType.FP64 ? "fp64" : "fp32";
        String opName = reduction.op().name();
        return "reduce_"
                + opName
                + "_axis_"
                + reduction.axis()
                + (reduction.keepDims() ? "_keep" : "_drop")
                + "_"
                + suffix
                + "_"
                + Integer.toHexString(hash).replace('-', '0');
    }

    record KernelProgramSpec(String name, String source) {}
}
