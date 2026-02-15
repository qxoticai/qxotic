package ai.qxotic.jota.runtime.javaaot;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.tir.*;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class JavaAotComputeEngine implements ComputeEngine {

    private final JavaAotMemoryDomain memoryDomain;

    JavaAotComputeEngine(JavaAotMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
    }

    @Override
    public Device device() {
        return Device.JAVA_AOT;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        try {
            return executeFastPath(graph, inputs);
        } catch (UnsupportedOperationException e) {
            return executeFallback(graph, inputs);
        }
    }

    private MemoryView<?> executeFastPath(TIRGraph graph, List<Tensor> inputs) {
        Evaluator evaluator = new Evaluator(graph, inputs);
        List<TIRNode> outputs = graph.outputs();
        if (outputs.isEmpty()) {
            throw new IllegalArgumentException("Graph has no outputs");
        }
        return evaluator.evaluate(outputs.getFirst());
    }

    private MemoryView<?> executeFallback(TIRGraph graph, List<Tensor> inputs) {
        List<MemoryView<?>> materialized = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            materialized.add(toJavaAot(input.materialize()));
        }
        List<MemoryView<MemorySegment>> outputs =
                TIRInterpreter.execute(graph, materialized, memoryDomain);
        return toJavaAot(outputs.getFirst());
    }

    private final class Evaluator {
        private final MemoryAccess<MemorySegment> access;
        private final Map<TIRNode, MemoryView<MemorySegment>> boundInputs = new IdentityHashMap<>();
        private final Map<TIRNode, MemoryView<MemorySegment>> cache = new IdentityHashMap<>();

        Evaluator(TIRGraph graph, List<Tensor> inputs) {
            @SuppressWarnings("unchecked")
            MemoryAccess<MemorySegment> typed =
                    (MemoryAccess<MemorySegment>) memoryDomain.directAccess();
            this.access = typed;
            int inputIndex = 0;
            for (TIRNode node : graph.inputs()) {
                if (node instanceof IotaConstant) {
                    continue;
                }
                if (inputIndex >= inputs.size()) {
                    throw new IllegalArgumentException(
                            "Missing input tensor for graph input: " + node);
                }
                boundInputs.put(node, toJavaAot(inputs.get(inputIndex++).materialize()));
            }
        }

        MemoryView<MemorySegment> evaluate(TIRNode node) {
            MemoryView<MemorySegment> cached = cache.get(node);
            if (cached != null) {
                return cached;
            }
            MemoryView<MemorySegment> result =
                    switch (node) {
                        case TensorInput input -> requireInput(input);
                        case ScalarInput input -> requireInput(input);
                        case ScalarConstant constant ->
                                constant(constant.rawBits(), constant.dataType(), constant.shape());
                        case IotaConstant iota -> iota(iota);
                        case UnaryOp unary -> unary(unary);
                        case BinaryOp binary -> binary(binary);
                        case TernaryOp ternary -> ternary(ternary);
                        case CastOp cast -> cast(cast);
                        case GatherOp gather ->
                                throw new UnsupportedOperationException("gather fallback");
                        case ViewTransform view -> view(view);
                        case Contiguous contiguous -> contiguous(contiguous);
                        case ReductionOp reduction -> reduction(reduction);
                    };
            cache.put(node, result);
            return result;
        }

        private MemoryView<MemorySegment> reduction(ReductionOp reduction) {
            MemoryView<MemorySegment> input = evaluate(reduction.input());
            MemoryView<MemorySegment> output = allocate(reduction.dataType(), reduction.shape());
            if (!input.isRowMajorContiguous() || !output.isRowMajorContiguous()) {
                throw new UnsupportedOperationException("non-contiguous reduction fallback");
            }
            if (input.dataType() != reduction.dataType()) {
                throw new UnsupportedOperationException("mixed-type reduction fallback");
            }

            int rank = input.shape().flatRank();
            int[] axes = normalizeAxes(reduction.axes(), rank);

            long reducedElements = reducedElementCount(input.shape(), axes);
            long outputElements = output.shape().size();
            long inBase = input.byteOffset();
            long outBase = output.byteOffset();
            long inStep = input.dataType().byteSize();
            long outStep = output.dataType().byteSize();
            DataType type = reduction.dataType();

            if (areSuffixAxes(rank, axes)) {
                for (long outIdx = 0; outIdx < outputElements; outIdx++) {
                    long inOffset = inBase + outIdx * reducedElements * inStep;
                    long outOffset = outBase + outIdx * outStep;
                    reduceContiguousChunk(
                            reduction.op(),
                            type,
                            input.memory(),
                            inOffset,
                            reducedElements,
                            inStep,
                            output.memory(),
                            outOffset);
                }
                return output;
            }

            long[] inputStrides = rowMajorElementStrides(input.shape());
            int[] reducedAxes = axes;
            int[] nonReducedAxes = complementAxes(rank, reducedAxes);
            long[] reducedStrides = new long[reducedAxes.length];
            long[] reducedDims = new long[reducedAxes.length];
            for (int i = 0; i < reducedAxes.length; i++) {
                int axis = reducedAxes[i];
                reducedStrides[i] = inputStrides[axis];
                reducedDims[i] = input.shape().flatAt(axis);
            }

            long[] outputDims = flatDims(output.shape());
            boolean keepDims = output.shape().flatRank() == rank;
            long[] outCoord = new long[outputDims.length];
            long[] redCoord = new long[reducedDims.length];

            for (long outIdx = 0; outIdx < outputElements; outIdx++) {
                long outOffset = outBase + outIdx * outStep;
                decodeLinear(outIdx, outputDims, outCoord);

                long baseElems = 0;
                if (keepDims) {
                    for (int axis : nonReducedAxes) {
                        baseElems += outCoord[axis] * inputStrides[axis];
                    }
                } else {
                    for (int i = 0; i < nonReducedAxes.length; i++) {
                        baseElems += outCoord[i] * inputStrides[nonReducedAxes[i]];
                    }
                }

                long inOffset = inBase + baseElems * inStep;

                reduceContiguousGather(
                        reduction.op(),
                        type,
                        input.memory(),
                        inStep,
                        inOffset,
                        reducedElements,
                        reducedDims,
                        reducedStrides,
                        redCoord,
                        output.memory(),
                        outOffset);
            }
            return output;
        }

        private MemoryView<MemorySegment> requireInput(TIRNode node) {
            MemoryView<MemorySegment> value = boundInputs.get(node);
            if (value == null) {
                throw new IllegalArgumentException("Missing bound input for node: " + node);
            }
            return value;
        }

        private MemoryView<MemorySegment> iota(IotaConstant iota) {
            MemoryView<MemorySegment> output = allocate(iota.dataType(), iota.shape());
            long offset = output.byteOffset();
            long step = iota.dataType().byteSize();
            for (long i = 0; i < iota.shape().size(); i++) {
                writeFromLong(output.memory(), offset, iota.dataType(), i);
                offset += step;
            }
            return output;
        }

        private MemoryView<MemorySegment> unary(UnaryOp unary) {
            MemoryView<MemorySegment> input = evaluate(unary.input());
            MemoryView<MemorySegment> output = allocate(unary.dataType(), unary.shape());
            if (!input.shape().equals(output.shape())) {
                throw new UnsupportedOperationException("unary broadcast fallback");
            }
            if (input.isRowMajorContiguous() && output.isRowMajorContiguous()) {
                long inOffset = input.byteOffset();
                long outOffset = output.byteOffset();
                long inStep = input.dataType().byteSize();
                long outStep = output.dataType().byteSize();
                for (long i = 0; i < output.shape().size(); i++) {
                    applyUnary(
                            unary.op(),
                            input.memory(),
                            inOffset,
                            output.memory(),
                            outOffset,
                            unary.dataType());
                    inOffset += inStep;
                    outOffset += outStep;
                }
                return output;
            }
            for (long i = 0; i < output.shape().size(); i++) {
                long inOffset = Indexing.linearToOffset(input, i);
                long outOffset = Indexing.linearToOffset(output, i);
                applyUnary(
                        unary.op(),
                        input.memory(),
                        inOffset,
                        output.memory(),
                        outOffset,
                        unary.dataType());
            }
            return output;
        }

        private MemoryView<MemorySegment> binary(BinaryOp binary) {
            MemoryView<MemorySegment> left = evaluate(binary.left());
            MemoryView<MemorySegment> right = evaluate(binary.right());
            MemoryView<MemorySegment> output = allocate(binary.dataType(), binary.shape());
            if (!left.shape().equals(output.shape()) || !right.shape().equals(output.shape())) {
                throw new UnsupportedOperationException("binary broadcast fallback");
            }
            if (left.isRowMajorContiguous()
                    && right.isRowMajorContiguous()
                    && output.isRowMajorContiguous()) {
                long leftOffset = left.byteOffset();
                long rightOffset = right.byteOffset();
                long outOffset = output.byteOffset();
                long leftStep = left.dataType().byteSize();
                long rightStep = right.dataType().byteSize();
                long outStep = output.dataType().byteSize();
                for (long i = 0; i < output.shape().size(); i++) {
                    applyBinary(
                            binary.op(),
                            left.memory(),
                            leftOffset,
                            right.memory(),
                            rightOffset,
                            output.memory(),
                            outOffset,
                            binary.left().dataType(),
                            binary.dataType());
                    leftOffset += leftStep;
                    rightOffset += rightStep;
                    outOffset += outStep;
                }
                return output;
            }
            for (long i = 0; i < output.shape().size(); i++) {
                long leftOffset = Indexing.linearToOffset(left, i);
                long rightOffset = Indexing.linearToOffset(right, i);
                long outOffset = Indexing.linearToOffset(output, i);
                applyBinary(
                        binary.op(),
                        left.memory(),
                        leftOffset,
                        right.memory(),
                        rightOffset,
                        output.memory(),
                        outOffset,
                        binary.left().dataType(),
                        binary.dataType());
            }
            return output;
        }

        private MemoryView<MemorySegment> ternary(TernaryOp ternary) {
            if (ternary.op() != TernaryOperator.WHERE) {
                throw new UnsupportedOperationException("Unsupported ternary op: " + ternary.op());
            }
            MemoryView<MemorySegment> cond = evaluate(ternary.cond());
            MemoryView<MemorySegment> onTrue = evaluate(ternary.trueExpr());
            MemoryView<MemorySegment> onFalse = evaluate(ternary.falseExpr());
            MemoryView<MemorySegment> output = allocate(ternary.dataType(), ternary.shape());
            if (!cond.shape().equals(output.shape())
                    || !onTrue.shape().equals(output.shape())
                    || !onFalse.shape().equals(output.shape())) {
                throw new UnsupportedOperationException("where broadcast fallback");
            }
            if (cond.isRowMajorContiguous()
                    && onTrue.isRowMajorContiguous()
                    && onFalse.isRowMajorContiguous()
                    && output.isRowMajorContiguous()) {
                long condOffset = cond.byteOffset();
                long trueOffset = onTrue.byteOffset();
                long falseOffset = onFalse.byteOffset();
                long outOffset = output.byteOffset();
                long condStep = cond.dataType().byteSize();
                long valueStep = onTrue.dataType().byteSize();
                long outStep = output.dataType().byteSize();
                for (long i = 0; i < output.shape().size(); i++) {
                    boolean pickTrue = access.readByte(cond.memory(), condOffset) != 0;
                    copyScalar(
                            pickTrue ? onTrue.memory() : onFalse.memory(),
                            pickTrue ? trueOffset : falseOffset,
                            output.memory(),
                            outOffset,
                            ternary.dataType());
                    condOffset += condStep;
                    trueOffset += valueStep;
                    falseOffset += valueStep;
                    outOffset += outStep;
                }
                return output;
            }
            for (long i = 0; i < output.shape().size(); i++) {
                long condOffset = Indexing.linearToOffset(cond, i);
                long trueOffset = Indexing.linearToOffset(onTrue, i);
                long falseOffset = Indexing.linearToOffset(onFalse, i);
                long outOffset = Indexing.linearToOffset(output, i);
                boolean pickTrue = access.readByte(cond.memory(), condOffset) != 0;
                copyScalar(
                        pickTrue ? onTrue.memory() : onFalse.memory(),
                        pickTrue ? trueOffset : falseOffset,
                        output.memory(),
                        outOffset,
                        ternary.dataType());
            }
            return output;
        }

        private MemoryView<MemorySegment> cast(CastOp cast) {
            MemoryView<MemorySegment> input = evaluate(cast.input());
            MemoryView<MemorySegment> output = allocate(cast.targetDataType(), cast.shape());
            if (!input.shape().equals(output.shape())) {
                throw new UnsupportedOperationException("cast broadcast fallback");
            }
            if (input.isRowMajorContiguous() && output.isRowMajorContiguous()) {
                long inOffset = input.byteOffset();
                long outOffset = output.byteOffset();
                long inStep = input.dataType().byteSize();
                long outStep = output.dataType().byteSize();
                for (long i = 0; i < output.shape().size(); i++) {
                    castScalar(
                            input.memory(),
                            inOffset,
                            cast.input().dataType(),
                            output.memory(),
                            outOffset,
                            cast.targetDataType());
                    inOffset += inStep;
                    outOffset += outStep;
                }
                return output;
            }
            for (long i = 0; i < output.shape().size(); i++) {
                long inOffset = Indexing.linearToOffset(input, i);
                long outOffset = Indexing.linearToOffset(output, i);
                castScalar(
                        input.memory(),
                        inOffset,
                        cast.input().dataType(),
                        output.memory(),
                        outOffset,
                        cast.targetDataType());
            }
            return output;
        }

        private MemoryView<MemorySegment> view(ViewTransform view) {
            if (view.needsLazyIndexing()) {
                throw new UnsupportedOperationException("lazy view fallback");
            }
            MemoryView<MemorySegment> input = evaluate(view.input());
            return MemoryView.of(
                    input.memory(), input.byteOffset(), input.dataType(), view.layout());
        }

        private MemoryView<MemorySegment> contiguous(Contiguous contiguous) {
            MemoryView<MemorySegment> input = evaluate(contiguous.input());
            if (input.isRowMajorContiguous()) {
                return input;
            }
            MemoryView<MemorySegment> output = allocate(contiguous.dataType(), contiguous.shape());
            for (long i = 0; i < output.shape().size(); i++) {
                long inOffset = Indexing.linearToOffset(input, i);
                long outOffset = Indexing.linearToOffset(output, i);
                copyScalar(input.memory(), inOffset, output.memory(), outOffset, input.dataType());
            }
            return output;
        }

        private MemoryView<MemorySegment> constant(long rawBits, DataType dataType, Shape shape) {
            MemoryView<MemorySegment> output = allocate(dataType, shape);
            long offset = output.byteOffset();
            long step = dataType.byteSize();
            for (long i = 0; i < shape.size(); i++) {
                writeRawBits(output.memory(), offset, dataType, rawBits);
                offset += step;
            }
            return output;
        }

        private void reduceContiguousChunk(
                ReductionOperator op,
                DataType type,
                Memory<MemorySegment> input,
                long inOffset,
                long count,
                long inStep,
                Memory<MemorySegment> output,
                long outOffset) {
            if (type == DataType.BOOL) {
                int acc =
                        switch (op) {
                            case SUM -> 0;
                            case PROD, MIN -> 1;
                            case MAX -> 0;
                        };
                long offset = inOffset;
                for (long i = 0; i < count; i++) {
                    int v = access.readByte(input, offset) != 0 ? 1 : 0;
                    acc =
                            switch (op) {
                                case SUM -> acc + v;
                                case PROD, MIN -> acc & v;
                                case MAX -> acc | v;
                            };
                    offset += inStep;
                }
                access.writeByte(output, outOffset, (byte) acc);
                return;
            }

            if (type.isFloatingPoint()) {
                double acc =
                        switch (op) {
                            case SUM -> 0.0;
                            case PROD -> 1.0;
                            case MIN -> Double.POSITIVE_INFINITY;
                            case MAX -> Double.NEGATIVE_INFINITY;
                        };
                long offset = inOffset;
                for (long i = 0; i < count; i++) {
                    double v = readAsDouble(input, offset, type);
                    acc =
                            switch (op) {
                                case SUM -> acc + v;
                                case PROD -> acc * v;
                                case MIN -> Math.min(acc, v);
                                case MAX -> Math.max(acc, v);
                            };
                    offset += inStep;
                }
                writeFromDouble(output, outOffset, type, acc);
                return;
            }

            long acc =
                    switch (op) {
                        case SUM -> 0L;
                        case PROD -> 1L;
                        case MIN -> Long.MAX_VALUE;
                        case MAX -> Long.MIN_VALUE;
                    };
            long offset = inOffset;
            for (long i = 0; i < count; i++) {
                long v = readAsLong(input, offset, type);
                acc =
                        switch (op) {
                            case SUM -> acc + v;
                            case PROD -> acc * v;
                            case MIN -> Math.min(acc, v);
                            case MAX -> Math.max(acc, v);
                        };
                offset += inStep;
            }
            writeFromLong(output, outOffset, type, acc);
        }

        private void reduceContiguousGather(
                ReductionOperator op,
                DataType type,
                Memory<MemorySegment> input,
                long inStep,
                long inOffset,
                long count,
                long[] reducedDims,
                long[] reducedStrides,
                long[] redCoord,
                Memory<MemorySegment> output,
                long outOffset) {
            if (type == DataType.BOOL) {
                int acc =
                        switch (op) {
                            case SUM -> 0;
                            case PROD, MIN -> 1;
                            case MAX -> 0;
                        };
                for (long i = 0; i < count; i++) {
                    long offset =
                            gatherOffset(
                                    i, reducedDims, reducedStrides, redCoord, inOffset, inStep);
                    int v = access.readByte(input, offset) != 0 ? 1 : 0;
                    acc =
                            switch (op) {
                                case SUM -> acc + v;
                                case PROD, MIN -> acc & v;
                                case MAX -> acc | v;
                            };
                }
                access.writeByte(output, outOffset, (byte) acc);
                return;
            }

            if (type.isFloatingPoint()) {
                double acc =
                        switch (op) {
                            case SUM -> 0.0;
                            case PROD -> 1.0;
                            case MIN -> Double.POSITIVE_INFINITY;
                            case MAX -> Double.NEGATIVE_INFINITY;
                        };
                for (long i = 0; i < count; i++) {
                    long offset =
                            gatherOffset(
                                    i, reducedDims, reducedStrides, redCoord, inOffset, inStep);
                    double v = readAsDouble(input, offset, type);
                    acc =
                            switch (op) {
                                case SUM -> acc + v;
                                case PROD -> acc * v;
                                case MIN -> Math.min(acc, v);
                                case MAX -> Math.max(acc, v);
                            };
                }
                writeFromDouble(output, outOffset, type, acc);
                return;
            }

            long acc =
                    switch (op) {
                        case SUM -> 0L;
                        case PROD -> 1L;
                        case MIN -> Long.MAX_VALUE;
                        case MAX -> Long.MIN_VALUE;
                    };
            for (long i = 0; i < count; i++) {
                long offset =
                        gatherOffset(i, reducedDims, reducedStrides, redCoord, inOffset, inStep);
                long v = readAsLong(input, offset, type);
                acc =
                        switch (op) {
                            case SUM -> acc + v;
                            case PROD -> acc * v;
                            case MIN -> Math.min(acc, v);
                            case MAX -> Math.max(acc, v);
                        };
            }
            writeFromLong(output, outOffset, type, acc);
        }

        private long gatherOffset(
                long linear,
                long[] reducedDims,
                long[] reducedStrides,
                long[] redCoord,
                long baseOffset,
                long inStep) {
            decodeLinear(linear, reducedDims, redCoord);
            long elemOffset = 0;
            for (int i = 0; i < redCoord.length; i++) {
                elemOffset += redCoord[i] * reducedStrides[i];
            }
            return baseOffset + elemOffset * inStep;
        }

        private void decodeLinear(long linear, long[] dims, long[] outCoord) {
            for (int i = dims.length - 1; i >= 0; i--) {
                long dim = dims[i];
                outCoord[i] = linear % dim;
                linear /= dim;
            }
        }

        private long[] flatDims(Shape shape) {
            int rank = shape.flatRank();
            long[] dims = new long[rank];
            for (int i = 0; i < rank; i++) {
                dims[i] = shape.flatAt(i);
            }
            return dims;
        }

        private long[] rowMajorElementStrides(Shape shape) {
            int rank = shape.flatRank();
            long[] strides = new long[rank];
            long stride = 1;
            for (int axis = rank - 1; axis >= 0; axis--) {
                strides[axis] = stride;
                stride *= shape.flatAt(axis);
            }
            return strides;
        }

        private int[] complementAxes(int rank, int[] reducedAxes) {
            int[] nonReduced = new int[rank - reducedAxes.length];
            int reducedCursor = 0;
            int out = 0;
            for (int axis = 0; axis < rank; axis++) {
                if (reducedCursor < reducedAxes.length && reducedAxes[reducedCursor] == axis) {
                    reducedCursor++;
                } else {
                    nonReduced[out++] = axis;
                }
            }
            return nonReduced;
        }

        private int[] normalizeAxes(int[] axes, int rank) {
            int[] normalized = new int[axes.length];
            for (int i = 0; i < axes.length; i++) {
                int axis = axes[i];
                if (axis < 0) {
                    axis += rank;
                }
                if (axis < 0 || axis >= rank) {
                    throw new IllegalArgumentException("invalid reduction axis: " + axes[i]);
                }
                normalized[i] = axis;
            }
            Arrays.sort(normalized);
            for (int i = 1; i < normalized.length; i++) {
                if (normalized[i - 1] == normalized[i]) {
                    throw new IllegalArgumentException(
                            "duplicate reduction axis: " + normalized[i]);
                }
            }
            return normalized;
        }

        private boolean areSuffixAxes(int rank, int[] axes) {
            int first = rank - axes.length;
            for (int i = 0; i < axes.length; i++) {
                if (axes[i] != first + i) {
                    return false;
                }
            }
            return true;
        }

        private long reducedElementCount(Shape shape, int[] axes) {
            long size = 1;
            for (int axis : axes) {
                size *= shape.flatAt(axis);
            }
            return size;
        }
    }

    private MemoryView<MemorySegment> allocate(DataType dataType, Shape shape) {
        return MemoryView.of(
                memoryDomain.memoryAllocator().allocateMemory(dataType, shape),
                dataType,
                Layout.rowMajor(shape));
    }

    @SuppressWarnings("unchecked")
    private MemoryView<MemorySegment> toJavaAot(MemoryView<?> view) {
        if (view.memory().device().equals(Device.JAVA_AOT)) {
            return (MemoryView<MemorySegment>) view;
        }
        MemoryView<MemorySegment> dst = allocate(view.dataType(), view.shape());
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryDomain.copy(srcDomain, srcView, memoryDomain, dst);
        return dst;
    }

    private void copyScalar(
            Memory<MemorySegment> src,
            long srcOffset,
            Memory<MemorySegment> dst,
            long dstOffset,
            DataType type) {
        switch (type.name()) {
            case "bool", "i8" ->
                    access().writeByte(dst, dstOffset, access().readByte(src, srcOffset));
            case "i16", "fp16", "bf16" ->
                    access().writeShort(dst, dstOffset, access().readShort(src, srcOffset));
            case "i32", "fp32" ->
                    access().writeInt(dst, dstOffset, access().readInt(src, srcOffset));
            case "i64", "fp64" ->
                    access().writeLong(dst, dstOffset, access().readLong(src, srcOffset));
            default -> throw new UnsupportedOperationException("Unsupported type: " + type);
        }
    }

    private void writeRawBits(Memory<MemorySegment> dst, long offset, DataType type, long rawBits) {
        if (type == DataType.BOOL || type == DataType.I8) {
            access().writeByte(dst, offset, (byte) rawBits);
            return;
        }
        if (type == DataType.I16 || type == DataType.FP16 || type == DataType.BF16) {
            access().writeShort(dst, offset, (short) rawBits);
            return;
        }
        if (type == DataType.I32 || type == DataType.FP32) {
            access().writeInt(dst, offset, (int) rawBits);
            return;
        }
        if (type == DataType.I64 || type == DataType.FP64) {
            access().writeLong(dst, offset, rawBits);
            return;
        }
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private void writeFromLong(Memory<MemorySegment> dst, long offset, DataType type, long value) {
        if (type == DataType.BOOL || type == DataType.I8) {
            access().writeByte(dst, offset, (byte) value);
            return;
        }
        if (type == DataType.I16) {
            access().writeShort(dst, offset, (short) value);
            return;
        }
        if (type == DataType.I32) {
            access().writeInt(dst, offset, (int) value);
            return;
        }
        if (type == DataType.I64) {
            access().writeLong(dst, offset, value);
            return;
        }
        if (type == DataType.FP32) {
            access().writeFloat(dst, offset, value);
            return;
        }
        if (type == DataType.FP64) {
            access().writeDouble(dst, offset, value);
            return;
        }
        throw new UnsupportedOperationException("Unsupported iota type: " + type);
    }

    private void castScalar(
            Memory<MemorySegment> src,
            long srcOffset,
            DataType srcType,
            Memory<MemorySegment> dst,
            long dstOffset,
            DataType dstType) {
        if (dstType == DataType.BOOL) {
            access().writeByte(
                            dst,
                            dstOffset,
                            (byte) (readAsDouble(src, srcOffset, srcType) != 0.0 ? 1 : 0));
            return;
        }
        if (dstType == DataType.I8) {
            access().writeByte(dst, dstOffset, (byte) readAsLong(src, srcOffset, srcType));
            return;
        }
        if (dstType == DataType.I16) {
            access().writeShort(dst, dstOffset, (short) readAsLong(src, srcOffset, srcType));
            return;
        }
        if (dstType == DataType.I32) {
            access().writeInt(dst, dstOffset, (int) readAsLong(src, srcOffset, srcType));
            return;
        }
        if (dstType == DataType.I64) {
            access().writeLong(dst, dstOffset, readAsLong(src, srcOffset, srcType));
            return;
        }
        if (dstType == DataType.FP16) {
            access().writeShort(
                            dst,
                            dstOffset,
                            Float.floatToFloat16((float) readAsDouble(src, srcOffset, srcType)));
            return;
        }
        if (dstType == DataType.BF16) {
            access().writeShort(
                            dst,
                            dstOffset,
                            BFloat16.fromFloat((float) readAsDouble(src, srcOffset, srcType)));
            return;
        }
        if (dstType == DataType.FP32) {
            access().writeFloat(dst, dstOffset, (float) readAsDouble(src, srcOffset, srcType));
            return;
        }
        if (dstType == DataType.FP64) {
            access().writeDouble(dst, dstOffset, readAsDouble(src, srcOffset, srcType));
            return;
        }
        throw new UnsupportedOperationException("Unsupported cast target type: " + dstType);
    }

    private void applyUnary(
            UnaryOperator op,
            Memory<MemorySegment> src,
            long srcOffset,
            Memory<MemorySegment> dst,
            long dstOffset,
            DataType type) {
        double x = readAsDouble(src, srcOffset, type);
        switch (op) {
            case NEGATE -> writeFromDouble(dst, dstOffset, type, -x);
            case ABS -> writeFromDouble(dst, dstOffset, type, Math.abs(x));
            case EXP -> writeFromDouble(dst, dstOffset, type, Math.exp(x));
            case LOG -> writeFromDouble(dst, dstOffset, type, Math.log(x));
            case SQRT -> writeFromDouble(dst, dstOffset, type, Math.sqrt(x));
            case SQUARE -> writeFromDouble(dst, dstOffset, type, x * x);
            case SIN -> writeFromDouble(dst, dstOffset, type, Math.sin(x));
            case COS -> writeFromDouble(dst, dstOffset, type, Math.cos(x));
            case TAN -> writeFromDouble(dst, dstOffset, type, Math.tan(x));
            case TANH -> writeFromDouble(dst, dstOffset, type, Math.tanh(x));
            case RECIPROCAL -> writeFromDouble(dst, dstOffset, type, 1.0 / x);
            case LOGICAL_NOT -> access().writeByte(dst, dstOffset, (byte) (x == 0.0 ? 1 : 0));
            case BITWISE_NOT ->
                    writeFromLong(dst, dstOffset, type, ~readAsLong(src, srcOffset, type));
        }
    }

    private void applyBinary(
            BinaryOperator op,
            Memory<MemorySegment> left,
            long leftOffset,
            Memory<MemorySegment> right,
            long rightOffset,
            Memory<MemorySegment> dst,
            long dstOffset,
            DataType inputType,
            DataType outputType) {
        double l = readAsDouble(left, leftOffset, inputType);
        double r = readAsDouble(right, rightOffset, inputType);
        switch (op) {
            case ADD -> writeFromDouble(dst, dstOffset, outputType, l + r);
            case SUBTRACT -> writeFromDouble(dst, dstOffset, outputType, l - r);
            case MULTIPLY -> writeFromDouble(dst, dstOffset, outputType, l * r);
            case DIVIDE -> writeFromDouble(dst, dstOffset, outputType, l / r);
            case MIN -> writeFromDouble(dst, dstOffset, outputType, Math.min(l, r));
            case MAX -> writeFromDouble(dst, dstOffset, outputType, Math.max(l, r));
            case POW -> writeFromDouble(dst, dstOffset, outputType, Math.pow(l, r));
            case LOGICAL_AND ->
                    access().writeByte(dst, dstOffset, (byte) ((l != 0.0 && r != 0.0) ? 1 : 0));
            case LOGICAL_OR ->
                    access().writeByte(dst, dstOffset, (byte) ((l != 0.0 || r != 0.0) ? 1 : 0));
            case LOGICAL_XOR ->
                    access().writeByte(dst, dstOffset, (byte) (((l != 0.0) ^ (r != 0.0)) ? 1 : 0));
            case BITWISE_AND ->
                    writeFromLong(
                            dst,
                            dstOffset,
                            outputType,
                            readAsLong(left, leftOffset, inputType)
                                    & readAsLong(right, rightOffset, inputType));
            case BITWISE_OR ->
                    writeFromLong(
                            dst,
                            dstOffset,
                            outputType,
                            readAsLong(left, leftOffset, inputType)
                                    | readAsLong(right, rightOffset, inputType));
            case BITWISE_XOR ->
                    writeFromLong(
                            dst,
                            dstOffset,
                            outputType,
                            readAsLong(left, leftOffset, inputType)
                                    ^ readAsLong(right, rightOffset, inputType));
            case EQUAL ->
                    access().writeByte(dst, dstOffset, (byte) (Double.compare(l, r) == 0 ? 1 : 0));
            case LESS_THAN -> access().writeByte(dst, dstOffset, (byte) (l < r ? 1 : 0));
        }
    }

    private long readAsLong(Memory<MemorySegment> memory, long offset, DataType type) {
        if (type == DataType.BOOL || type == DataType.I8) {
            return access().readByte(memory, offset);
        }
        if (type == DataType.I16) {
            return access().readShort(memory, offset);
        }
        if (type == DataType.I32) {
            return access().readInt(memory, offset);
        }
        if (type == DataType.I64) {
            return access().readLong(memory, offset);
        }
        if (type == DataType.FP16) {
            return (long) Float.float16ToFloat(access().readShort(memory, offset));
        }
        if (type == DataType.BF16) {
            return (long) BFloat16.toFloat(access().readShort(memory, offset));
        }
        if (type == DataType.FP32) {
            return (long) access().readFloat(memory, offset);
        }
        if (type == DataType.FP64) {
            return (long) access().readDouble(memory, offset);
        }
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private double readAsDouble(Memory<MemorySegment> memory, long offset, DataType type) {
        if (type == DataType.BOOL || type == DataType.I8) {
            return access().readByte(memory, offset);
        }
        if (type == DataType.I16) {
            return access().readShort(memory, offset);
        }
        if (type == DataType.I32) {
            return access().readInt(memory, offset);
        }
        if (type == DataType.I64) {
            return access().readLong(memory, offset);
        }
        if (type == DataType.FP16) {
            return Float.float16ToFloat(access().readShort(memory, offset));
        }
        if (type == DataType.BF16) {
            return BFloat16.toFloat(access().readShort(memory, offset));
        }
        if (type == DataType.FP32) {
            return access().readFloat(memory, offset);
        }
        if (type == DataType.FP64) {
            return access().readDouble(memory, offset);
        }
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private void writeFromDouble(
            Memory<MemorySegment> dst, long offset, DataType type, double value) {
        if (type == DataType.BOOL) {
            access().writeByte(dst, offset, (byte) (value != 0 ? 1 : 0));
            return;
        }
        if (type == DataType.I8) {
            access().writeByte(dst, offset, (byte) value);
            return;
        }
        if (type == DataType.I16) {
            access().writeShort(dst, offset, (short) value);
            return;
        }
        if (type == DataType.I32) {
            access().writeInt(dst, offset, (int) value);
            return;
        }
        if (type == DataType.I64) {
            access().writeLong(dst, offset, (long) value);
            return;
        }
        if (type == DataType.FP16) {
            access().writeShort(dst, offset, Float.floatToFloat16((float) value));
            return;
        }
        if (type == DataType.BF16) {
            access().writeShort(dst, offset, BFloat16.fromFloat((float) value));
            return;
        }
        if (type == DataType.FP32) {
            access().writeFloat(dst, offset, (float) value);
            return;
        }
        if (type == DataType.FP64) {
            access().writeDouble(dst, offset, value);
            return;
        }
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private MemoryAccess<MemorySegment> access() {
        @SuppressWarnings("unchecked")
        MemoryAccess<MemorySegment> typed =
                (MemoryAccess<MemorySegment>) memoryDomain.directAccess();
        return typed;
    }
}
