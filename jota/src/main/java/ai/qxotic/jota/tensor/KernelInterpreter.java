package ai.qxotic.jota.tensor;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import static ai.qxotic.jota.DataType.*;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

public final class KernelInterpreter {

    private KernelInterpreter() {}

    public static void execute(
            ExpressionGraph graph,
            MemoryView<MemorySegment>[] inputs,
            MemoryView<MemorySegment> output) {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(output, "output");

        long[] outputShape = output.shape().toArray();
        long[] outputStride = output.byteStride().toArray();
        long outputBaseOffset = output.byteOffset();
        MemorySegment outputBase = output.memory().base();

        InputAccessor[] accessors = new InputAccessor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            MemoryView<MemorySegment> view = inputs[i];
            accessors[i] =
                    new InputAccessor(
                            view.memory().base(),
                            view.byteOffset(),
                            view.shape().toArray(),
                            view.byteStride().toArray());
        }

        long size = output.shape().size();
        ExprNode root = graph.root();
        if (root.dataType() == DataType.FP32) {
            for (long index = 0; index < size; index++) {
                float value = evalFloat(root, index, accessors);
                long offset = offsetForIndex(index, outputBaseOffset, outputShape, outputStride);
                outputBase.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);
            }
            return;
        }
        if (root.dataType() == DataType.I32) {
            for (long index = 0; index < size; index++) {
                int value = evalInt(root, index, accessors);
                long offset = offsetForIndex(index, outputBaseOffset, outputShape, outputStride);
                outputBase.set(ValueLayout.JAVA_INT_UNALIGNED, offset, value);
            }
            return;
        }
        if (root.dataType() == DataType.BOOL) {
            for (long index = 0; index < size; index++) {
                byte value = evalBool(root, index, accessors);
                long offset = offsetForIndex(index, outputBaseOffset, outputShape, outputStride);
                outputBase.set(ValueLayout.JAVA_BYTE, offset, value);
            }
            return;
        }
        throw new IllegalStateException("Unsupported output type: " + root.dataType());
    }

    private static float evalFloat(ExprNode node, long index, InputAccessor[] inputs) {
        if (node instanceof InputNode input) {
            return inputs[input.index()].readFloat(index);
        }
        if (node instanceof ScalarNode scalar) {
            return scalar.value().floatValue();
        }
        if (node instanceof UnaryNode unary) {
            float value = evalFloat(unary.input(), index, inputs);
            return applyUnaryFloat(unary.op(), value);
        }
        if (node instanceof BinaryNode binary) {
            float left = evalFloat(binary.left(), index, inputs);
            float right = evalFloat(binary.right(), index, inputs);
            return applyBinaryFloat(binary.op(), left, right);
        }
        if (node instanceof CastNode cast) {
            ExprNode input = cast.input();
            if (input.dataType() == DataType.I32) {
                return (float) evalInt(input, index, inputs);
            }
            return evalFloat(input, index, inputs);
        }
        if (node instanceof TernaryNode ternary) {
            byte cond = evalBool(ternary.condition(), index, inputs);
            return cond == 0
                    ? evalFloat(ternary.falseValue(), index, inputs)
                    : evalFloat(ternary.trueValue(), index, inputs);
        }
        if (node instanceof ReductionNode reduction) {
            return evalReductionFloat(reduction, index, inputs);
        }
        throw new IllegalStateException("Unsupported node: " + node);
    }

    private static int evalInt(ExprNode node, long index, InputAccessor[] inputs) {
        if (node instanceof InputNode input) {
            return inputs[input.index()].readInt(index);
        }
        if (node instanceof ScalarNode scalar) {
            return scalar.value().intValue();
        }
        if (node instanceof UnaryNode unary) {
            int value = evalInt(unary.input(), index, inputs);
            return applyUnaryInt(unary.op(), value);
        }
        if (node instanceof BinaryNode binary) {
            int left = evalInt(binary.left(), index, inputs);
            int right = evalInt(binary.right(), index, inputs);
            return applyBinaryInt(binary.op(), left, right);
        }
        if (node instanceof CastNode cast) {
            ExprNode input = cast.input();
            if (input.dataType() == DataType.FP32) {
                return (int) evalFloat(input, index, inputs);
            }
            return evalInt(input, index, inputs);
        }
        if (node instanceof TernaryNode ternary) {
            byte cond = evalBool(ternary.condition(), index, inputs);
            return cond == 0
                    ? evalInt(ternary.falseValue(), index, inputs)
                    : evalInt(ternary.trueValue(), index, inputs);
        }
        if (node instanceof ReductionNode reduction) {
            return evalReductionInt(reduction, index, inputs);
        }
        throw new IllegalStateException("Unsupported node: " + node);
    }

    private static byte evalBool(ExprNode node, long index, InputAccessor[] inputs) {
        if (node instanceof InputNode input) {
            return inputs[input.index()].readByte(index);
        }
        if (node instanceof ScalarNode scalar) {
            return (byte) (scalar.value().intValue() == 0 ? 0 : 1);
        }
        if (node instanceof UnaryNode unary) {
            byte value = evalBool(unary.input(), index, inputs);
            return applyUnaryBool(unary.op(), value);
        }
        if (node instanceof BinaryNode binary) {
            return applyBinaryBool(binary, index, inputs);
        }
        if (node instanceof CastNode cast) {
            ExprNode input = cast.input();
            if (input.dataType() == DataType.I32) {
                return (byte) (evalInt(input, index, inputs) == 0 ? 0 : 1);
            }
            if (input.dataType() == DataType.FP32) {
                return (byte) (evalFloat(input, index, inputs) == 0.0f ? 0 : 1);
            }
            return evalBool(input, index, inputs);
        }
        if (node instanceof TernaryNode ternary) {
            byte cond = evalBool(ternary.condition(), index, inputs);
            return cond == 0
                    ? evalBool(ternary.falseValue(), index, inputs)
                    : evalBool(ternary.trueValue(), index, inputs);
        }
        throw new IllegalStateException("Unsupported node: " + node);
    }

    private static float applyUnaryFloat(UnaryOp op, float value) {
        return switch (op.name()) {
            case "negate" -> -value;
            case "abs" -> Math.abs(value);
            case "exp" -> (float) Math.exp(value);
            case "log" -> (float) Math.log(value);
            case "sqrt" -> (float) Math.sqrt(value);
            case "square" -> value * value;
            case "sin" -> (float) Math.sin(value);
            case "cos" -> (float) Math.cos(value);
            case "tanh" -> (float) Math.tanh(value);
            case "sigmoid" -> 1.0f / (1.0f + (float) Math.exp(-value));
            case "relu" -> Math.max(0.0f, value);
            case "gelu" -> {
                float cubic = value * value * value;
                float inner = 0.79788456f * (value + 0.044715f * cubic);
                yield 0.5f * value * (1.0f + (float) Math.tanh(inner));
            }
            case "silu" -> value / (1.0f + (float) Math.exp(-value));
            default -> throw new IllegalStateException("Unsupported unary op: " + op.name());
        };
    }

    private static byte applyUnaryBool(UnaryOp op, byte value) {
        return switch (op.name()) {
            case "logicalNot" -> (byte) (value == 0 ? 1 : 0);
            default -> throw new IllegalStateException("Unsupported unary op: " + op.name());
        };
    }

    private static byte applyBinaryBool(BinaryNode binary, long index, InputAccessor[] inputs) {
        return switch (binary.op().name()) {
            case "logicalAnd" ->
                    (byte)
                            ((evalBool(binary.left(), index, inputs) != 0
                                            && evalBool(binary.right(), index, inputs) != 0)
                                    ? 1
                                    : 0);
            case "logicalOr" ->
                    (byte)
                            ((evalBool(binary.left(), index, inputs) != 0
                                            || evalBool(binary.right(), index, inputs) != 0)
                                    ? 1
                                    : 0);
            case "logicalXor" ->
                    (byte)
                            (((evalBool(binary.left(), index, inputs) != 0)
                                            ^ (evalBool(binary.right(), index, inputs) != 0))
                                    ? 1
                                    : 0);
            case "equal", "lessThan" ->
                    (byte)
                            (compareValues(binary, index, inputs) ? 1 : 0);
            default ->
                    throw new IllegalStateException("Unsupported binary op: " + binary.op().name());
        };
    }

    private static boolean compareValues(
            BinaryNode binary, long index, InputAccessor[] inputs) {
        ExprNode left = binary.left();
        ExprNode right = binary.right();
        DataType type = left.dataType();
        if (type != right.dataType()) {
            throw new IllegalStateException("Mismatched comparison types: " + type + " vs " + right.dataType());
        }
        boolean equals = "equal".equals(binary.op().name());
        if (type == DataType.BOOL) {
            byte leftValue = evalBool(left, index, inputs);
            byte rightValue = evalBool(right, index, inputs);
            return equals ? leftValue == rightValue : leftValue < rightValue;
        }
        if (type == DataType.I32) {
            int leftValue = evalInt(left, index, inputs);
            int rightValue = evalInt(right, index, inputs);
            return equals ? leftValue == rightValue : leftValue < rightValue;
        }
        if (type == DataType.FP32) {
            float leftValue = evalFloat(left, index, inputs);
            float rightValue = evalFloat(right, index, inputs);
            return equals ? leftValue == rightValue : leftValue < rightValue;
        }
        if (type.isIntegral()) {
            long leftValue = evalIntegralValue(left, type, index, inputs);
            long rightValue = evalIntegralValue(right, type, index, inputs);
            return equals ? leftValue == rightValue : leftValue < rightValue;
        }
        if (type.isFloatingPoint()) {
            double leftValue = evalFloatingValue(left, type, index, inputs);
            double rightValue = evalFloatingValue(right, type, index, inputs);
            return equals ? leftValue == rightValue : leftValue < rightValue;
        }
        throw new IllegalStateException("Unsupported comparison type: " + type);
    }

    private static long evalIntegralValue(
            ExprNode node, DataType type, long index, InputAccessor[] inputs) {
        if (node instanceof InputNode input) {
            if (type == DataType.BOOL || type == DataType.I8) {
                return inputs[input.index()].readByte(index);
            }
            if (type == DataType.I16) {
                return inputs[input.index()].readShort(index);
            }
            if (type == DataType.I32) {
                return inputs[input.index()].readInt(index);
            }
            if (type == DataType.I64) {
                return inputs[input.index()].readLong(index);
            }
            throw new IllegalStateException("Unsupported integral type: " + type);
        }
        if (node instanceof ScalarNode scalar) {
            return scalar.value().longValue();
        }
        if (node instanceof CastNode cast) {
            if (type == DataType.I32) {
                return evalInt(cast.input(), index, inputs);
            }
            if (type == DataType.I64) {
                return (long) evalInt(cast.input(), index, inputs);
            }
            return evalIntegralValue(cast.input(), type, index, inputs);
        }
        throw new IllegalStateException("Unsupported node for integral comparison: " + node);
    }

    private static double evalFloatingValue(
            ExprNode node, DataType type, long index, InputAccessor[] inputs) {
        if (node instanceof InputNode input) {
            if (type == DataType.FP16) {
                return Float.float16ToFloat(inputs[input.index()].readShort(index));
            }
            if (type == DataType.BF16) {
                return BFloat16.toFloat(inputs[input.index()].readShort(index));
            }
            if (type == DataType.FP32) {
                return inputs[input.index()].readFloat(index);
            }
            if (type == DataType.FP64) {
                return inputs[input.index()].readDouble(index);
            }
            throw new IllegalStateException("Unsupported floating type: " + type);
        }
        if (node instanceof ScalarNode scalar) {
            return scalar.value().doubleValue();
        }
        if (node instanceof CastNode cast) {
            if (type == DataType.FP32) {
                return evalFloat(cast.input(), index, inputs);
            }
            if (type == DataType.FP64) {
                return evalFloat(cast.input(), index, inputs);
            }
            return evalFloatingValue(cast.input(), type, index, inputs);
        }
        throw new IllegalStateException("Unsupported node for floating comparison: " + node);
    }

    private static int applyUnaryInt(UnaryOp op, int value) {
        return switch (op.name()) {
            case "negate" -> -value;
            case "abs" -> Math.abs(value);
            case "square" -> value * value;
            case "relu" -> Math.max(0, value);
            default ->
                    throw new IllegalStateException("Unsupported unary op for I32: " + op.name());
        };
    }

    private static float applyBinaryFloat(BinaryOp op, float left, float right) {
        return switch (op.name()) {
            case "add" -> left + right;
            case "subtract" -> left - right;
            case "multiply" -> left * right;
            case "divide" -> left / right;
            case "min" -> Math.min(left, right);
            case "max" -> Math.max(left, right);
            case "pow" -> (float) Math.pow(left, right);
            default -> throw new IllegalStateException("Unsupported binary op: " + op.name());
        };
    }

    private static int applyBinaryInt(BinaryOp op, int left, int right) {
        return switch (op.name()) {
            case "add" -> left + right;
            case "subtract" -> left - right;
            case "multiply" -> left * right;
            case "divide" -> left / right;
            case "min" -> Math.min(left, right);
            case "max" -> Math.max(left, right);
            case "pow" -> (int) Math.pow(left, right);
            default -> throw new IllegalStateException("Unsupported binary op: " + op.name());
        };
    }

    private static float evalReductionFloat(
            ReductionNode reduction, long index, InputAccessor[] inputs) {
        ReductionInfo info = collectReductionInfo(reduction);
        if (info.dataType() != DataType.FP32) {
            throw new IllegalStateException(
                    "Unsupported reduction output type: " + info.dataType());
        }
        Shape inShape = info.input().layout().shape();
        Shape outShape = reduction.layout().shape();
        long[] inDims = inShape.toArray();
        long[] outCoord = Indexing.linearToCoord(outShape, index);
        long[] inCoord = new long[inDims.length];
        if (info.keepDims()) {
            System.arraycopy(outCoord, 0, inCoord, 0, inCoord.length);
        } else {
            int outDim = 0;
            for (int dim = 0; dim < inCoord.length; dim++) {
                boolean reduced = false;
                for (int axis : info.axes()) {
                    if (axis == dim) {
                        reduced = true;
                        break;
                    }
                }
                inCoord[dim] = reduced ? 0 : outCoord[outDim++];
            }
        }
        long[] reduceDims = new long[info.axes().length];
        for (int i = 0; i < info.axes().length; i++) {
            reduceDims[i] = inDims[info.axes()[i]];
        }
        Shape reduceShape = Shape.flat(reduceDims);
        long reduceSize = reduceShape.size();
        if (info.op() == ReductionOp.SUM) {
            float acc = 0.0f;
            for (long r = 0; r < reduceSize; r++) {
                long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
                for (int j = 0; j < info.axes().length; j++) {
                    inCoord[info.axes()[j]] = reduceCoord[j];
                }
                long inputIndex = Indexing.coordToLinear(inShape, inCoord);
                acc += evalFloat(info.input(), inputIndex, inputs);
            }
            return acc;
        }
        if (info.op() == ReductionOp.PROD) {
            float acc = 1.0f;
            for (long r = 0; r < reduceSize; r++) {
                long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
                for (int j = 0; j < info.axes().length; j++) {
                    inCoord[info.axes()[j]] = reduceCoord[j];
                }
                long inputIndex = Indexing.coordToLinear(inShape, inCoord);
                acc *= evalFloat(info.input(), inputIndex, inputs);
            }
            return acc;
        }
        if (reduceSize == 0) {
            return 0.0f;
        }
        for (int j = 0; j < info.axes().length; j++) {
            inCoord[info.axes()[j]] = 0;
        }
        long startIndex = Indexing.coordToLinear(inShape, inCoord);
        float acc = evalFloat(info.input(), startIndex, inputs);
        for (long r = 1; r < reduceSize; r++) {
            long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
            for (int j = 0; j < info.axes().length; j++) {
                inCoord[info.axes()[j]] = reduceCoord[j];
            }
            long inputIndex = Indexing.coordToLinear(inShape, inCoord);
            float value = evalFloat(info.input(), inputIndex, inputs);
            if (info.op() == ReductionOp.MIN) {
                acc = Math.min(acc, value);
            } else {
                acc = Math.max(acc, value);
            }
        }
        return acc;
    }

    private static int evalReductionInt(
            ReductionNode reduction, long index, InputAccessor[] inputs) {
        ReductionInfo info = collectReductionInfo(reduction);
        if (info.dataType() != DataType.I32) {
            throw new IllegalStateException(
                    "Unsupported reduction output type: " + info.dataType());
        }
        Shape inShape = info.input().layout().shape();
        Shape outShape = reduction.layout().shape();
        long[] inDims = inShape.toArray();
        long[] outCoord = Indexing.linearToCoord(outShape, index);
        long[] inCoord = new long[inDims.length];
        if (info.keepDims()) {
            System.arraycopy(outCoord, 0, inCoord, 0, inCoord.length);
        } else {
            int outDim = 0;
            for (int dim = 0; dim < inCoord.length; dim++) {
                boolean reduced = false;
                for (int axis : info.axes()) {
                    if (axis == dim) {
                        reduced = true;
                        break;
                    }
                }
                inCoord[dim] = reduced ? 0 : outCoord[outDim++];
            }
        }
        long[] reduceDims = new long[info.axes().length];
        for (int i = 0; i < info.axes().length; i++) {
            reduceDims[i] = inDims[info.axes()[i]];
        }
        Shape reduceShape = Shape.flat(reduceDims);
        long reduceSize = reduceShape.size();
        if (info.op() == ReductionOp.SUM) {
            int acc = 0;
            for (long r = 0; r < reduceSize; r++) {
                long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
                for (int j = 0; j < info.axes().length; j++) {
                    inCoord[info.axes()[j]] = reduceCoord[j];
                }
                long inputIndex = Indexing.coordToLinear(inShape, inCoord);
                acc += evalInt(info.input(), inputIndex, inputs);
            }
            return acc;
        }
        if (info.op() == ReductionOp.PROD) {
            int acc = 1;
            for (long r = 0; r < reduceSize; r++) {
                long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
                for (int j = 0; j < info.axes().length; j++) {
                    inCoord[info.axes()[j]] = reduceCoord[j];
                }
                long inputIndex = Indexing.coordToLinear(inShape, inCoord);
                acc *= evalInt(info.input(), inputIndex, inputs);
            }
            return acc;
        }
        if (reduceSize == 0) {
            return 0;
        }
        for (int j = 0; j < info.axes().length; j++) {
            inCoord[info.axes()[j]] = 0;
        }
        long startIndex = Indexing.coordToLinear(inShape, inCoord);
        int acc = evalInt(info.input(), startIndex, inputs);
        for (long r = 1; r < reduceSize; r++) {
            long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);
            for (int j = 0; j < info.axes().length; j++) {
                inCoord[info.axes()[j]] = reduceCoord[j];
            }
            long inputIndex = Indexing.coordToLinear(inShape, inCoord);
            int value = evalInt(info.input(), inputIndex, inputs);
            if (info.op() == ReductionOp.MIN) {
                acc = Math.min(acc, value);
            } else {
                acc = Math.max(acc, value);
            }
        }
        return acc;
    }

    private static ReductionInfo collectReductionInfo(ReductionNode reduction) {
        ArrayList<Integer> axes = new ArrayList<>();
        ExprNode current = reduction;
        ReductionOp op = reduction.op();
        boolean keepDims = reduction.keepDims();
        while (current instanceof ReductionNode node
                && node.op() == op
                && node.keepDims() == keepDims) {
            axes.add(node.axis());
            current = node.input();
        }
        int[] axisArray = axes.stream().distinct().mapToInt(Integer::intValue).toArray();
        Arrays.sort(axisArray);
        return new ReductionInfo(current, axisArray, keepDims, op, reduction.dataType());
    }

    private record ReductionInfo(
            ExprNode input, int[] axes, boolean keepDims, ReductionOp op, DataType dataType) {}

    private static long offsetForIndex(long index, long baseOffset, long[] shape, long[] stride) {
        long offset = baseOffset;
        long remaining = index;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            long size = shape[dim];
            if (size == 0) {
                return baseOffset;
            }
            long coord = remaining % size;
            remaining /= size;
            offset += coord * stride[dim];
        }
        return offset;
    }

    private record InputAccessor(MemorySegment base, long baseOffset, long[] shape, long[] stride) {

        byte readByte(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_BYTE, offset);
        }

        short readShort(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        }

        int readInt(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        }

        long readLong(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        }

        float readFloat(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
        }

        double readDouble(long index) {
            long offset = offsetForIndex(index, baseOffset, shape, stride);
            return base.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
        }
    }
}
