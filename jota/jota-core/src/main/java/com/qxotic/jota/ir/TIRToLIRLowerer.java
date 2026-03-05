package com.qxotic.jota.ir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Util;
import com.qxotic.jota.ir.lir.*;
import com.qxotic.jota.ir.lir.LIRCanonicalizerPass;
import com.qxotic.jota.ir.tir.*;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Lowers TIR (Tensor IR) graphs to LIR (Loop-level IR) graphs.
 *
 * <p>TIR represents tensor operations semantically, while LIR makes loops and memory access
 * explicit.
 */
public class TIRToLIRLowerer implements TIRVisitor<LIRExprNode> {

    private final Map<TIRNode, BufferRef> inputBuffers = new IdentityHashMap<>();
    private final Map<TIRNode, com.qxotic.jota.ir.lir.ScalarInput> inputScalars =
            new IdentityHashMap<>();
    private final Map<TIRNode, BufferRef> outputBuffers = new IdentityHashMap<>();
    private final Map<TensorInput, Layout> inputLayoutOverrides = new IdentityHashMap<>();
    private final List<LIRInput> allInputs = new ArrayList<>();
    private final List<BufferRef> allOutputs = new ArrayList<>();
    private final IdentityHashMap<TIRNode, LIRExprNode> scalarCache = new IdentityHashMap<>();
    private int nextId = 0;
    private int nextScalarId = 0;
    private boolean scalarCachingEnabled = false;
    private LIRExprGraph exprGraph;

    // Current loop indices for the output iteration
    private final List<LIRExprNode> loopIndices = new ArrayList<>();

    /** Lowers a TIRGraph to an LIRGraph. */
    public LIRGraph lower(TIRGraph tirGraph) {
        return lower(tirGraph, null);
    }

    /**
     * Lowers a TIRGraph to an LIRGraph using optional runtime layout overrides for tensor inputs.
     */
    public LIRGraph lower(TIRGraph tirGraph, List<Layout> inputLayouts) {
        inputBuffers.clear();
        inputScalars.clear();
        outputBuffers.clear();
        inputLayoutOverrides.clear();
        allInputs.clear();
        allOutputs.clear();
        loopIndices.clear();
        scalarCache.clear();
        scalarCachingEnabled = false;
        nextId = 0;
        nextScalarId = 0;
        LIRExprGraph exprGraph = new LIRExprGraph();

        if (inputLayouts != null) {
            if (inputLayouts.size() != tirGraph.inputs().size()) {
                throw new IllegalArgumentException(
                        "Expected "
                                + tirGraph.inputs().size()
                                + " input layouts but got "
                                + inputLayouts.size());
            }
            for (int i = 0; i < tirGraph.inputs().size(); i++) {
                TIRNode input = tirGraph.inputs().get(i);
                if (input instanceof TensorInput tensorInput) {
                    Layout override = inputLayouts.get(i);
                    if (override != null) {
                        inputLayoutOverrides.put(tensorInput, override);
                    }
                }
            }
        }

        // Create inputs for materialized inputs
        // - IotaConstant: always virtual (computed from loop index)
        // - ScalarConstant as input: becomes a scalar parameter (dynamic, passed by value)
        // - ViewTransform wrapping ScalarConstant: also becomes a scalar parameter
        // - ScalarConstant not as input: inlined as constant (static)
        // - TensorInput: becomes a buffer reference (passed by pointer)
        for (TIRNode input : tirGraph.inputs()) {
            if (isVirtualInput(input)) {
                // IotaConstant is computed from loop index, no input needed
                continue;
            }

            if (input instanceof com.qxotic.jota.ir.tir.ScalarInput scalarInput) {
                com.qxotic.jota.ir.lir.ScalarInput lirInput =
                        new com.qxotic.jota.ir.lir.ScalarInput(nextId++, scalarInput.dataType());
                inputScalars.put(input, lirInput);
                allInputs.add(lirInput);
                continue;
            }

            // Check if this is a ScalarConstant or a ViewTransform chain wrapping one
            ScalarConstant underlyingScalar = extractUnderlyingScalarConstant(input);
            if (underlyingScalar != null) {
                // Scalar inputs become scalar parameters (passed by value)
                com.qxotic.jota.ir.lir.ScalarInput scalarInput =
                        new com.qxotic.jota.ir.lir.ScalarInput(
                                nextId++, underlyingScalar.dataType());
                // Map both the input node and the underlying scalar to this ScalarInput
                inputScalars.put(input, scalarInput);
                inputScalars.put(underlyingScalar, scalarInput);
                allInputs.add(scalarInput);
            } else {
                BufferRef buf = createBufferRef(input);
                inputBuffers.put(input, buf);
                allInputs.add(buf);
            }
        }

        // Create output buffers
        for (TIRNode output : tirGraph.outputs()) {
            BufferRef buf = createOutputBufferRef(output);
            outputBuffers.put(output, buf);
            allOutputs.add(buf);
        }

        // Generate loop body for each output
        List<LIRExprNode> statements = new ArrayList<>();
        for (int i = 0; i < tirGraph.outputs().size(); i++) {
            TIRNode output = tirGraph.outputs().get(i);
            BufferRef outBuf = allOutputs.get(i);
            LIRExprNode stmt = lowerOutput(exprGraph, output, outBuf);
            statements.add(stmt);
        }

        LIRExprNode body =
                statements.size() == 1 ? statements.getFirst() : exprGraph.block(statements);

        LIRGraph graph = new LIRGraph(exprGraph, allInputs, allOutputs, body);
        return new LIRCanonicalizerPass().run(graph);
    }

    /**
     * Returns true if the node is a "virtual" input that doesn't need backing memory. Virtual
     * inputs are computed on-the-fly from loop indices, rather than loaded from memory.
     *
     * <p>Note: ScalarConstant is NOT virtual when it's an explicit graph input - in that case it
     * becomes a scalar parameter that can vary between kernel invocations. Only IotaConstant is
     * always virtual since its values are determined purely by shape/index.
     */
    private boolean isVirtualInput(TIRNode node) {
        return node instanceof IotaConstant || node instanceof RandomUniformOp;
    }

    /**
     * Extracts the underlying ScalarConstant from a node if it's a ScalarConstant or a chain of
     * ViewTransforms wrapping a ScalarConstant.
     *
     * @param node the TIR node to examine
     * @return the underlying ScalarConstant, or null if not found
     */
    private ScalarConstant extractUnderlyingScalarConstant(TIRNode node) {
        if (node instanceof ScalarConstant sc) {
            return sc;
        }
        if (node instanceof ViewTransform vt) {
            return extractUnderlyingScalarConstant(vt.input());
        }
        return null;
    }

    private BufferRef createBufferRef(TIRNode node) {
        return BufferRef.of(nextId++, node.dataType(), layoutForNode(node));
    }

    private BufferRef createOutputBufferRef(TIRNode node) {
        // Outputs are always fresh allocations, so use row-major layout.
        // This avoids pathological strides from transpose/slice operations being
        // baked into the output buffer, which would waste memory.
        return BufferRef.of(nextId++, node.dataType(), Layout.rowMajor(node.shape()));
    }

    private Layout layoutForNode(TIRNode node) {
        if (node instanceof TensorInput input) {
            Layout override = inputLayoutOverrides.get(input);
            if (override != null) {
                return override;
            }
            return input.layout();
        }
        if (node instanceof ViewTransform view) {
            return view.layout();
        }
        return Layout.rowMajor(node.shape());
    }

    /** Computes byte strides from a layout and data type. */
    private long[] toByteStrides(Layout layout, DataType dtype) {
        int rank = (int) layout.shape().flatRank();
        long[] strides = new long[rank];
        long byteSize = dtype.byteSize();
        for (int i = 0; i < rank; i++) {
            strides[i] = layout.stride().flatAt(i) * byteSize;
        }
        return strides;
    }

    /** Lowers a single output expression to LIR loop nest. */
    private LIRExprNode lowerOutput(LIRExprGraph exprGraph, TIRNode output, BufferRef outBuf) {
        this.exprGraph = exprGraph;
        // Pure reductions (output is directly a ReductionOp) are handled here
        if (output instanceof ReductionOp reduction) {
            return lowerReduction(exprGraph, reduction, outBuf, null);
        }

        ReductionOp reductionRoot = findReductionRoot(output);
        if (reductionRoot != null) {
            LIRExprNode postOpValue =
                    buildPostOpExpr(
                            output,
                            reductionRoot,
                            exprGraph.scalarRef("acc0", reductionRoot.dataType()));
            return lowerReduction(exprGraph, reductionRoot, outBuf, postOpValue);
        }

        Layout outLayout = layoutForNode(output).flatten();
        int rank = outLayout.shape().flatRank();

        // Create loop index variables
        loopIndices.clear();
        for (int i = 0; i < rank; i++) {
            loopIndices.add(exprGraph.indexVar("i" + i));
        }

        // Compute the scalar expression for this output
        beginScalarCaching();
        LIRExprNode value = output.accept(this);
        endScalarCaching();

        // Compute output offset
        LIRExprNode outOffset = computeOffset(loopIndices, outBuf.byteStrides());

        // Create the store
        Store store = exprGraph.store(outBuf, outOffset, value);
        LIRExprNode storeNode = store;

        // Wrap in nested loops (innermost first)
        LIRExprNode body = ensureYield(exprGraph, storeNode);
        for (int i = rank - 1; i >= 0; i--) {
            long bound = outLayout.shape().flatAt(i);
            body =
                    exprGraph.structuredFor(
                            ((IVar) loopIndices.get(i)).name(),
                            exprGraph.indexConst(0),
                            exprGraph.indexConst(bound),
                            exprGraph.indexConst(1),
                            List.of(),
                            ensureYield(exprGraph, body));
        }

        return body;
    }

    private void beginScalarCaching() {
        scalarCache.clear();
        nextScalarId = 0;
        scalarCachingEnabled = true;
    }

    private void endScalarCaching() {
        scalarCachingEnabled = false;
        scalarCache.clear();
    }

    private LIRExprNode cacheScalar(TIRNode node, Supplier<LIRExprNode> builder) {
        if (!scalarCachingEnabled) {
            return builder.get();
        }
        LIRExprNode cached = scalarCache.get(node);
        if (cached != null) {
            return cached;
        }
        LIRExprNode value = builder.get();
        scalarCache.put(node, value);
        nextScalarId++;
        return value;
    }

    /** Lowers a reduction operation to LIR with accumulators. */
    private LIRExprNode lowerReduction(
            LIRExprGraph exprGraph, ReductionOp reduction, BufferRef outBuf) {
        return lowerReduction(exprGraph, reduction, outBuf, null);
    }

    private LIRExprNode lowerReduction(
            LIRExprGraph exprGraph,
            ReductionOp reduction,
            BufferRef outBuf,
            LIRExprNode postOpValue) {
        this.exprGraph = exprGraph;
        TIRNode input = reduction.input();
        Layout inputLayout = layoutForNode(input).flatten();
        int inputRank = inputLayout.shape().flatRank();
        int[] axes = reduction.axes();
        DataType dtype = reduction.dataType();
        ReductionOperator op = reduction.op();

        // Determine which dimensions are reduced
        boolean[] isReduced = new boolean[inputRank];
        for (int axis : axes) {
            isReduced[axis] = true;
        }

        // Create index variables for all input dimensions
        loopIndices.clear();
        List<LIRExprNode> outerIndices = new ArrayList<>();
        List<LIRExprNode> innerIndices = new ArrayList<>();
        List<Long> outerBounds = new ArrayList<>();
        List<Long> innerBounds = new ArrayList<>();

        for (int i = 0; i < inputRank; i++) {
            LIRExprNode idx = exprGraph.indexVar("i" + i);
            loopIndices.add(idx);
            if (isReduced[i]) {
                innerIndices.add(idx);
                innerBounds.add(inputLayout.shape().flatAt(i));
            } else {
                outerIndices.add(idx);
                outerBounds.add(inputLayout.shape().flatAt(i));
            }
        }

        // Load value from input and update accumulator (loop-carried)
        LIRExprNode inputValue = input.accept(this);

        LIRExprNode reductionBody =
                buildReductionLoop(exprGraph, innerIndices, innerBounds, dtype, op, inputValue);

        // Read accumulator and store to output (apply post-ops if provided)
        LIRExprNode read = exprGraph.scalarRef("acc0", dtype);
        LIRExprNode storeValue = (postOpValue != null) ? postOpValue : read;

        // Compute output offset from outer indices only
        LIRExprNode outOffset;
        if (outerIndices.isEmpty()) {
            outOffset = exprGraph.indexConst(0);
        } else {
            outOffset = computeOffset(outerIndices, outBuf.byteStrides());
        }
        Store store = exprGraph.store(outBuf, outOffset, storeValue);

        // Combine: declare acc, reduction loops, store result
        Block innerBlock = exprGraph.block(List.of(reductionBody, store));

        // Wrap in outer loops (over non-reduced dimensions)
        LIRExprNode body = innerBlock;
        for (int i = outerIndices.size() - 1; i >= 0; i--) {
            body =
                    exprGraph.structuredFor(
                            ((IVar) outerIndices.get(i)).name(),
                            exprGraph.indexConst(0),
                            exprGraph.indexConst(outerBounds.get(i)),
                            exprGraph.indexConst(1),
                            List.of(),
                            ensureYield(exprGraph, body));
        }

        return body;
    }

    private ReductionOp findReductionRoot(TIRNode node) {
        ReductionOp[] found = new ReductionOp[1];
        findReductionRootRec(node, found, new IdentityHashMap<>());
        return found[0];
    }

    private void findReductionRootRec(
            TIRNode node, ReductionOp[] found, IdentityHashMap<TIRNode, Boolean> visited) {
        if (visited.put(node, Boolean.TRUE) != null) {
            return;
        }
        if (node instanceof ReductionOp reduction) {
            if (found[0] != null && found[0] != reduction) {
                throw new UnsupportedOperationException(
                        "Multiple reductions in a single output are not supported");
            }
            found[0] = reduction;
            return;
        }
        switch (node) {
            case UnaryOp op -> findReductionRootRec(op.input(), found, visited);
            case BinaryOp op -> {
                findReductionRootRec(op.left(), found, visited);
                findReductionRootRec(op.right(), found, visited);
            }
            case TernaryOp op -> {
                findReductionRootRec(op.cond(), found, visited);
                findReductionRootRec(op.trueExpr(), found, visited);
                findReductionRootRec(op.falseExpr(), found, visited);
            }
            case CastOp op -> findReductionRootRec(op.input(), found, visited);
            case GatherOp op -> {
                findReductionRootRec(op.input(), found, visited);
                findReductionRootRec(op.indices(), found, visited);
            }
            case ViewTransform vt -> findReductionRootRec(vt.input(), found, visited);
            case Contiguous contig -> findReductionRootRec(contig.input(), found, visited);
            default -> {}
        }
    }

    private LIRExprNode buildPostOpExpr(
            TIRNode node, ReductionOp reduction, LIRExprNode reductionValue) {
        if (node == reduction) {
            return reductionValue;
        }
        switch (node) {
            case UnaryOp op -> {
                LIRExprNode input = buildPostOpExpr(op.input(), reduction, reductionValue);
                return exprGraph.scalarUnary(op.op(), input);
            }
            case BinaryOp op -> {
                LIRExprNode left = buildPostOpExpr(op.left(), reduction, reductionValue);
                LIRExprNode right = buildPostOpExpr(op.right(), reduction, reductionValue);
                return exprGraph.scalarBinary(op.op(), left, right);
            }
            case TernaryOp op -> {
                LIRExprNode cond = buildPostOpExpr(op.cond(), reduction, reductionValue);
                LIRExprNode trueVal = buildPostOpExpr(op.trueExpr(), reduction, reductionValue);
                LIRExprNode falseVal = buildPostOpExpr(op.falseExpr(), reduction, reductionValue);
                return exprGraph.scalarTernary(cond, trueVal, falseVal);
            }
            case CastOp op -> {
                LIRExprNode input = buildPostOpExpr(op.input(), reduction, reductionValue);
                return exprGraph.scalarCast(input, op.targetDataType());
            }
            case ScalarConstant sc -> {
                return visitScalarConstant(sc);
            }
            case ViewTransform __ ->
                    throw new UnsupportedOperationException(
                            "View transforms after reductions are not supported");
            case Contiguous __ ->
                    throw new UnsupportedOperationException(
                            "Contiguous post-ops after reductions are not supported");
            case ReductionOp __ ->
                    throw new UnsupportedOperationException(
                            "Nested reductions are not supported in post-ops");
            case GatherOp __ ->
                    throw new UnsupportedOperationException(
                            "Gather operations are not supported in post-ops after reductions");
            case TensorInput __ ->
                    throw new UnsupportedOperationException(
                            "Post-ops after reductions must be element-wise on the reduction result");
            case com.qxotic.jota.ir.tir.ScalarInput si -> {
                com.qxotic.jota.ir.lir.ScalarInput scalar = inputScalars.get(si);
                if (scalar == null) {
                    throw new IllegalStateException("Unknown ScalarInput: " + si.id());
                }
                return exprGraph.scalarInput(scalar.id(), scalar.dataType());
            }
            case IotaConstant __ ->
                    throw new UnsupportedOperationException(
                            "Post-ops after reductions must be element-wise on the reduction result");
            case RandomUniformOp __ ->
                    throw new UnsupportedOperationException(
                            "Post-ops after reductions must be element-wise on the reduction result");
        }
    }

    private LIRExprNode buildReductionLoop(
            LIRExprGraph exprGraph,
            List<LIRExprNode> innerIndices,
            List<Long> innerBounds,
            DataType dtype,
            ReductionOperator op,
            LIRExprNode inputValue) {
        int depth = innerIndices.size();
        if (depth == 0) {
            LIRExprNode combined =
                    combineAccumulator(
                            exprGraph, exprGraph.scalarRef("acc0", dtype), inputValue, op);
            return exprGraph.yield(List.of(combined));
        }

        String[] accNames = new String[depth];
        for (int i = 0; i < depth; i++) {
            accNames[i] = "acc" + i;
        }

        LIRExprNode identity = createIdentityLiteral(exprGraph, dtype, op);

        LIRExprNode combined =
                combineAccumulator(
                        exprGraph, exprGraph.scalarRef(accNames[depth - 1], dtype), inputValue, op);
        LIRExprNode inner = exprGraph.yield(List.of(combined));

        for (int level = depth - 1; level >= 0; level--) {
            String accName = accNames[level];
            LIRExprNode init =
                    (level == 0) ? identity : exprGraph.scalarRef(accNames[level - 1], dtype);

            Block body;
            if (level == depth - 1) {
                body = ensureYield(exprGraph, inner);
            } else {
                body =
                        exprGraph.block(
                                List.of(
                                        inner,
                                        exprGraph.yield(
                                                List.of(
                                                        exprGraph.scalarRef(
                                                                accNames[level + 1], dtype)))));
            }

            inner =
                    exprGraph.structuredFor(
                            ((IVar) innerIndices.get(level)).name(),
                            exprGraph.indexConst(0),
                            exprGraph.indexConst(innerBounds.get(level)),
                            exprGraph.indexConst(1),
                            List.of(new LoopIterArg(accName, dtype, init)),
                            body);
        }

        return inner;
    }

    private LIRExprNode combineAccumulator(
            LIRExprGraph exprGraph, LIRExprNode acc, LIRExprNode value, ReductionOperator op) {
        // Cast value to accumulator type if needed
        LIRExprNode castValue = value;
        if (value.dataType() != acc.dataType()) {
            castValue = exprGraph.scalarCast(value, acc.dataType());
        }
        return switch (op) {
            case SUM -> exprGraph.scalarBinary(BinaryOperator.ADD, acc, castValue);
            case PROD -> exprGraph.scalarBinary(BinaryOperator.MULTIPLY, acc, castValue);
            case MIN -> exprGraph.scalarBinary(BinaryOperator.MIN, acc, castValue);
            case MAX -> exprGraph.scalarBinary(BinaryOperator.MAX, acc, castValue);
        };
    }

    private LIRExprNode createIdentityLiteral(
            LIRExprGraph exprGraph, DataType dtype, ReductionOperator op) {
        return switch (op) {
            case SUM -> exprGraph.scalarConst(identityZeroBits(dtype), dtype);
            case PROD -> exprGraph.scalarConst(identityOneBits(dtype), dtype);
            case MIN -> exprGraph.scalarConst(identityMinBits(dtype), dtype);
            case MAX -> exprGraph.scalarConst(identityMaxBits(dtype), dtype);
        };
    }

    private long identityZeroBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(0.0f);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(0.0);
        }
        return 0L;
    }

    private long identityOneBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(1.0f);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(1.0);
        }
        return 1L;
    }

    private long identityMaxBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(Float.NEGATIVE_INFINITY);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY);
        } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
            // For FP16/BF16, use short representation of -infinity
            // In FP16, 0xFC00 represents negative infinity
            return (long) 0xFC00;
        } else if (dtype == DataType.I32) {
            return Integer.MIN_VALUE;
        } else if (dtype == DataType.I64) {
            return Long.MIN_VALUE;
        } else if (dtype == DataType.I8) {
            return Byte.MIN_VALUE;
        } else if (dtype == DataType.I16) {
            return Short.MIN_VALUE;
        }
        return Long.MIN_VALUE;
    }

    private long identityMinBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(Float.POSITIVE_INFINITY);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(Double.POSITIVE_INFINITY);
        } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
            // For FP16/BF16, use short representation of +infinity
            // In FP16, 0x7C00 represents positive infinity
            return 0x7C00;
        } else if (dtype == DataType.I32) {
            return Integer.MAX_VALUE;
        } else if (dtype == DataType.I64) {
            return Long.MAX_VALUE;
        } else if (dtype == DataType.I8) {
            return Byte.MAX_VALUE;
        } else if (dtype == DataType.I16) {
            return Short.MAX_VALUE;
        }
        return Long.MAX_VALUE;
    }

    private Block ensureYield(LIRExprGraph exprGraph, LIRExprNode body) {
        if (body instanceof Block block) {
            if (!block.statements().isEmpty() && block.statements().getLast() instanceof Yield) {
                return block;
            }
        }
        if (body instanceof Yield) {
            return exprGraph.block(List.of(body));
        }
        return exprGraph.block(List.of(body, exprGraph.yield(List.of())));
    }

    /** Computes byte offset from indices and byte strides. */
    private LIRExprNode computeOffset(List<? extends LIRExprNode> indices, long[] strides) {
        if (indices.isEmpty()) {
            return exprGraph.indexConst(0);
        }

        int indexRank = indices.size();
        int strideRank = strides.length;
        int strideShift = indexRank - strideRank;

        LIRExprNode offset = null;
        for (int i = 0; i < indexRank; i++) {
            int strideIndex = i - strideShift;
            long stride =
                    (strideIndex < 0 || strideIndex >= strideRank) ? 0L : strides[strideIndex];
            if (stride == 0) {
                continue; // Broadcasting dimension - skip
            }
            LIRExprNode term =
                    exprGraph.indexBinary(
                            IndexBinaryOp.MULTIPLY, indices.get(i), exprGraph.indexConst(stride));
            offset =
                    (offset == null)
                            ? term
                            : exprGraph.indexBinary(IndexBinaryOp.ADD, offset, term);
        }
        return offset != null ? offset : exprGraph.indexConst(0);
    }

    // ==================== TIR Visitor Methods ====================

    @Override
    public LIRExprNode visitTensorInput(TensorInput node) {
        return cacheScalar(
                node,
                () -> {
                    BufferRef buf = inputBuffers.get(node);
                    if (buf == null) {
                        throw new IllegalStateException("Unknown TensorInput: " + node.id());
                    }
                    LIRExprNode offset = computeOffset(loopIndices, buf.byteStrides());
                    return exprGraph.scalarLoad(buf, offset, buf.dataType());
                });
    }

    @Override
    public LIRExprNode visitScalarInput(com.qxotic.jota.ir.tir.ScalarInput node) {
        com.qxotic.jota.ir.lir.ScalarInput scalar = inputScalars.get(node);
        if (scalar == null) {
            throw new IllegalStateException("Unknown ScalarInput: " + node.id());
        }
        return exprGraph.scalarInput(scalar.id(), scalar.dataType());
    }

    @Override
    public LIRExprNode visitScalarConstant(ScalarConstant node) {
        // Check if this scalar is an explicit input (dynamic parameter)
        com.qxotic.jota.ir.lir.ScalarInput scalarInput = inputScalars.get(node);
        if (scalarInput != null) {
            // Scalar input - reference the scalar parameter directly (passed by value)
            return exprGraph.scalarInput(scalarInput.id(), scalarInput.dataType());
        }
        // Not an input - inline as literal
        return exprGraph.scalarConst(node.rawBits(), node.dataType());
    }

    @Override
    public LIRExprNode visitUnaryOp(UnaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    LIRExprNode input = node.input().accept(this);
                    return exprGraph.scalarUnary(node.op(), input);
                });
    }

    @Override
    public LIRExprNode visitBinaryOp(BinaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    LIRExprNode left = node.left().accept(this);
                    LIRExprNode right = node.right().accept(this);
                    return exprGraph.scalarBinary(node.op(), left, right);
                });
    }

    @Override
    public LIRExprNode visitTernaryOp(TernaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    LIRExprNode cond = node.cond().accept(this);
                    LIRExprNode trueVal = node.trueExpr().accept(this);
                    LIRExprNode falseVal = node.falseExpr().accept(this);
                    return exprGraph.scalarTernary(cond, trueVal, falseVal);
                });
    }

    @Override
    public LIRExprNode visitCastOp(CastOp node) {
        return cacheScalar(
                node,
                () -> {
                    LIRExprNode input = node.input().accept(this);
                    return exprGraph.scalarCast(input, node.targetDataType());
                });
    }

    @Override
    public LIRExprNode visitReductionOp(ReductionOp node) {
        // Reductions should have been handled in lowerOutput, not visited directly
        // This can happen if a reduction is used in a context that requires scalar evaluation
        // In such cases, we need to handle it - for now, throw a more helpful error
        throw new UnsupportedOperationException(
                "ReductionOp lowering not yet implemented - requires separate handling. "
                        + "Reductions with post-operations (e.g., .add(), .cast()) are not yet supported.");
    }

    @Override
    public LIRExprNode visitGatherOp(GatherOp node) {
        return cacheScalar(
                node,
                () -> {
                    int inputRank = (int) node.input().shape().flatRank();
                    int outputRank = (int) node.shape().flatRank();
                    int indicesRank = (int) node.indices().shape().flatRank();
                    int axis = Util.wrapAround(node.axis(), inputRank);
                    if (loopIndices.size() != outputRank) {
                        throw new IllegalStateException(
                                "Expected "
                                        + outputRank
                                        + " loop indices for gather output, got "
                                        + loopIndices.size());
                    }

                    List<LIRExprNode> indicesCoords = new ArrayList<>(indicesRank);
                    for (int i = 0; i < indicesRank; i++) {
                        indicesCoords.add(loopIndices.get(axis + i));
                    }

                    LIRExprNode gatheredIndexScalar =
                            evaluateAtIndices(node.indices(), indicesCoords);
                    LIRExprNode gatheredIndex = exprGraph.indexFromScalar(gatheredIndexScalar);

                    List<LIRExprNode> inputCoords = new ArrayList<>(inputRank);
                    for (int i = 0; i < axis; i++) {
                        inputCoords.add(loopIndices.get(i));
                    }
                    inputCoords.add(gatheredIndex);
                    for (int i = axis + 1; i < inputRank; i++) {
                        inputCoords.add(loopIndices.get(indicesRank + i - 1));
                    }

                    return evaluateAtIndices(node.input(), inputCoords);
                });
    }

    private LIRExprNode evaluateAtIndices(TIRNode node, List<LIRExprNode> indices) {
        List<LIRExprNode> savedIndices = new ArrayList<>(loopIndices);
        boolean savedCaching = scalarCachingEnabled;
        loopIndices.clear();
        loopIndices.addAll(indices);
        scalarCachingEnabled = false;
        try {
            return node.accept(this);
        } finally {
            loopIndices.clear();
            loopIndices.addAll(savedIndices);
            scalarCachingEnabled = savedCaching;
        }
    }

    @Override
    public LIRExprNode visitViewTransform(ViewTransform node) {
        return cacheScalar(
                node,
                () -> {
                    // Collect the chain of ViewTransforms
                    List<ViewTransform> chain = new ArrayList<>();
                    TIRNode current = node;
                    while (current instanceof ViewTransform vt) {
                        chain.add(0, vt); // prepend to get oldest-first order
                        current = vt.input();
                    }

                    // Handle the base input type (possibly wrapped in CastOp)
                    TIRNode baseNode = current;
                    DataType castType = null;

                    // Unwrap CastOp if present
                    if (baseNode instanceof CastOp castOp) {
                        castType = castOp.targetDataType();
                        baseNode = castOp.input();
                    }

                    LIRExprNode result;
                    if (baseNode instanceof TensorInput tensorInput) {
                        result = lowerTensorInputWithViewChain(tensorInput, chain, node);
                    } else if (baseNode instanceof com.qxotic.jota.ir.tir.ScalarInput scalarInput) {
                        com.qxotic.jota.ir.lir.ScalarInput scalar = inputScalars.get(scalarInput);
                        if (scalar == null) {
                            throw new IllegalStateException(
                                    "Unknown ScalarInput: " + scalarInput.id());
                        }
                        result = exprGraph.scalarInput(scalar.id(), scalar.dataType());
                    } else if (baseNode instanceof IotaConstant iotaConstant) {
                        result = lowerIotaConstantWithViewChain(iotaConstant, chain, node);
                    } else {
                        // For computed tensors, apply index remapping and evaluate input
                        List<LIRExprNode> remapped = new ArrayList<>(loopIndices);
                        for (int i = chain.size() - 1; i >= 0; i--) {
                            ViewTransform vt = chain.get(i);
                            remapped = applyInverseTransform(vt, remapped);
                        }
                        List<LIRExprNode> saved = new ArrayList<>(loopIndices);
                        loopIndices.clear();
                        loopIndices.addAll(remapped);
                        LIRExprNode remappedResult = baseNode.accept(this);
                        loopIndices.clear();
                        loopIndices.addAll(saved);
                        return remappedResult;
                    }

                    // Apply cast if needed
                    if (castType != null && castType != result.dataType()) {
                        return exprGraph.scalarCast(result, castType);
                    }
                    return result;
                });
    }

    private LIRExprNode lowerTensorInputWithViewChain(
            TensorInput tensorInput, List<ViewTransform> chain, ViewTransform node) {
        BufferRef buf = inputBuffers.get(tensorInput);
        if (buf == null) {
            throw new IllegalStateException("Unknown TensorInput: " + tensorInput.id());
        }

        // Compose index expressions by walking the chain in reverse from output space back to
        // input space. This is required for correctness even for view transforms that are not
        // marked lazy (e.g. reshape introducing singleton dimensions).
        List<LIRExprNode> indices = new ArrayList<>(loopIndices);

        // Apply transforms in REVERSE order (from output to input)
        for (int i = chain.size() - 1; i >= 0; i--) {
            ViewTransform vt = chain.get(i);
            indices = applyInverseTransform(vt, indices);
        }

        // Compute final offset using the BASE input's strides
        LIRExprNode offset = computeOffset(indices, buf.byteStrides());
        return exprGraph.scalarLoad(buf, offset, buf.dataType());
    }

    private LIRExprNode lowerIotaConstantWithViewChain(
            IotaConstant iotaConstant, List<ViewTransform> chain, ViewTransform node) {
        // For IotaConstant, we compute the linear index from the inverse-transformed coordinates
        // Start with output loop indices and transform back to IotaConstant's original space
        List<LIRExprNode> indices = new ArrayList<>(loopIndices);

        // Apply transforms in REVERSE order (from output to input)
        for (int i = chain.size() - 1; i >= 0; i--) {
            ViewTransform vt = chain.get(i);
            indices = applyInverseTransform(vt, indices);
        }

        // Now 'indices' are in IotaConstant's original flat space
        // Compute linear index from these coordinates using IotaConstant's layout
        Layout iotaLayout = Layout.rowMajor(iotaConstant.shape()).flatten();
        LIRExprNode linearIdx = null;
        long stride = 1;
        // Compute row-major linear index: sum(idx[i] * stride[i])
        for (int i = indices.size() - 1; i >= 0; i--) {
            LIRExprNode term =
                    exprGraph.indexBinary(
                            IndexBinaryOp.MULTIPLY, indices.get(i), exprGraph.indexConst(stride));
            linearIdx =
                    (linearIdx == null)
                            ? term
                            : exprGraph.indexBinary(IndexBinaryOp.ADD, term, linearIdx);
            stride *= (i < iotaLayout.shape().flatRank()) ? iotaLayout.shape().flatAt(i) : 1;
        }

        if (linearIdx == null) {
            linearIdx = exprGraph.indexConst(0);
        }

        // Cast index to the output type
        DataType targetType = iotaConstant.dataType();
        LIRExprNode indexValue = exprGraph.scalarFromIndex(linearIdx, DataType.I64);
        if (targetType == DataType.I64) {
            return indexValue;
        }
        return exprGraph.scalarCast(indexValue, targetType);
    }

    /**
     * Applies the inverse of a view transformation to convert output indices to input indices.
     *
     * <p>For example, if the transform is Transpose([1, 0]) on shape (A, B) -> (B, A), the inverse
     * maps (i, j) in output space to (j, i) in input space.
     */
    private List<LIRExprNode> applyInverseTransform(
            ViewTransform vt, List<LIRExprNode> outputIndices) {
        return switch (vt.kind()) {
            case ViewKind.Transpose transpose -> {
                // Transpose: permute indices using inverse permutation
                int[] invPerm = transpose.inverse();
                List<LIRExprNode> result = new ArrayList<>(outputIndices.size());
                for (int j : invPerm) {
                    result.add(outputIndices.get(j));
                }
                yield result;
            }
            case ViewKind.Reshape reshape -> {
                // Reshape: decompose linear index and recompose
                // First flatten output indices to linear index
                Shape toShape = reshape.toShape();
                Shape fromShape = reshape.fromShape();
                LIRExprNode linearIdx = flattenIndices(outputIndices, toShape);
                // Then decompose to input shape coordinates
                yield unflattenIndex(linearIdx, fromShape);
            }
            case ViewKind.Broadcast broadcast -> {
                // Broadcast: output has more/larger dims, input has fewer/smaller
                // For broadcast dims (input size 1), the input index is always 0
                // For other dims, pass through
                Shape fromShape = broadcast.fromShape();
                Shape toShape = broadcast.toShape();
                int fromRank = (int) fromShape.flatRank();
                int toRank = (int) toShape.flatRank();
                int offset = toRank - fromRank;

                List<LIRExprNode> result = new ArrayList<>();
                for (int i = 0; i < fromRank; i++) {
                    long fromDim = fromShape.flatAt(i);
                    if (fromDim == 1) {
                        // Broadcast dimension - input index is always 0
                        result.add(exprGraph.indexConst(0));
                    } else {
                        // Pass through
                        result.add(outputIndices.get(offset + i));
                    }
                }
                yield result;
            }
            case ViewKind.Expand expand -> {
                // Expand is similar to broadcast within same rank
                Shape fromShape = expand.fromShape();
                List<LIRExprNode> result = new ArrayList<>();
                for (int i = 0; i < fromShape.flatRank(); i++) {
                    long fromDim = fromShape.flatAt(i);
                    if (fromDim == 1) {
                        result.add(exprGraph.indexConst(0));
                    } else {
                        result.add(outputIndices.get(i));
                    }
                }
                yield result;
            }
            case ViewKind.Slice slice -> {
                // Slice: output index maps to input index with offset and step
                // input_idx = start + output_idx * step
                List<LIRExprNode> result = new ArrayList<>(outputIndices);
                LIRExprNode outIdx = outputIndices.get(slice.axis());
                LIRExprNode inIdx =
                        exprGraph.indexBinary(
                                IndexBinaryOp.ADD,
                                exprGraph.indexConst(slice.start()),
                                exprGraph.indexBinary(
                                        IndexBinaryOp.MULTIPLY,
                                        outIdx,
                                        exprGraph.indexConst(slice.step())));
                result.set(slice.axis(), inIdx);
                yield result;
            }
        };
    }

    /** Flattens multi-dimensional indices to a linear index using row-major order. */
    private LIRExprNode flattenIndices(List<LIRExprNode> indices, Shape shape) {
        if (indices.isEmpty()) {
            return exprGraph.indexConst(0);
        }
        LIRExprNode linear = null;
        long stride = 1;
        for (int i = (int) shape.flatRank() - 1; i >= 0; i--) {
            LIRExprNode term =
                    exprGraph.indexBinary(
                            IndexBinaryOp.MULTIPLY, indices.get(i), exprGraph.indexConst(stride));
            linear =
                    (linear == null)
                            ? term
                            : exprGraph.indexBinary(IndexBinaryOp.ADD, term, linear);
            stride *= shape.flatAt(i);
        }
        return linear != null ? linear : exprGraph.indexConst(0);
    }

    /** Decomposes a linear index into multi-dimensional indices using row-major order. */
    private List<LIRExprNode> unflattenIndex(LIRExprNode linearIdx, Shape shape) {
        int rank = (int) shape.flatRank();
        if (rank == 0) {
            return List.of();
        }
        List<LIRExprNode> indices = new ArrayList<>(rank);
        // Compute strides (row-major)
        long[] strides = new long[rank];
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape.flatAt(i + 1);
        }
        // Decompose: idx[i] = (linear / stride[i]) % dim[i]
        for (int i = 0; i < rank; i++) {
            LIRExprNode divided =
                    exprGraph.indexBinary(
                            IndexBinaryOp.DIVIDE, linearIdx, exprGraph.indexConst(strides[i]));
            LIRExprNode idx =
                    exprGraph.indexBinary(
                            IndexBinaryOp.MODULO, divided, exprGraph.indexConst(shape.flatAt(i)));
            indices.add(idx);
        }
        return indices;
    }

    @Override
    public LIRExprNode visitContiguous(Contiguous node) {
        // Contiguous forces a copy, but at this level we just read the input
        return cacheScalar(node, () -> node.input().accept(this));
    }

    @Override
    public LIRExprNode visitIotaConstant(IotaConstant node) {
        // Iota returns the index value - need to compute the linear index
        return cacheScalar(
                node,
                () -> {
                    if (loopIndices.isEmpty()) {
                        return exprGraph.scalarConst(0L, DataType.I64);
                    }

                    // Compute linear index from loop indices
                    Layout layout = Layout.rowMajor(node.shape()).flatten();
                    LIRExprNode linearIdx = null;
                    long stride = 1;
                    for (int i = layout.shape().flatRank() - 1; i >= 0; i--) {
                        LIRExprNode term =
                                exprGraph.indexBinary(
                                        IndexBinaryOp.MULTIPLY,
                                        loopIndices.get(i),
                                        exprGraph.indexConst(stride));
                        linearIdx =
                                (linearIdx == null)
                                        ? term
                                        : exprGraph.indexBinary(IndexBinaryOp.ADD, term, linearIdx);
                        stride *= layout.shape().flatAt(i);
                    }

                    // Cast index to the output type
                    if (linearIdx == null) {
                        linearIdx = exprGraph.indexConst(0);
                    }

                    // Index nodes are I64, cast to target type if needed
                    DataType targetType = node.dataType();
                    LIRExprNode indexValue = exprGraph.scalarFromIndex(linearIdx, DataType.I64);
                    if (targetType == DataType.I64) {
                        return indexValue;
                    }
                    return exprGraph.scalarCast(indexValue, targetType);
                });
    }

    @Override
    public LIRExprNode visitRandomUniformOp(RandomUniformOp node) {
        return cacheScalar(
                node,
                () -> {
                    int rank = node.shape().flatRank();
                    if (rank != loopIndices.size()) {
                        throw new IllegalStateException(
                                "RandomUniform rank/loop rank mismatch: "
                                        + rank
                                        + " vs "
                                        + loopIndices.size());
                    }

                    LIRExprNode linear = exprGraph.indexConst(0);
                    long stride = 1;
                    for (int i = rank - 1; i >= 0; i--) {
                        LIRExprNode term =
                                exprGraph.indexBinary(
                                        IndexBinaryOp.MULTIPLY,
                                        loopIndices.get(i),
                                        exprGraph.indexConst(stride));
                        linear = exprGraph.indexBinary(IndexBinaryOp.ADD, linear, term);
                        stride = Math.multiplyExact(stride, node.shape().flatAt(i));
                    }

                    long key0 = Math.floorMod((int) node.key0(), 1024);
                    long key1 = Math.floorMod((int) node.key1(), 1024);
                    LIRExprNode seed =
                            exprGraph.indexBinary(
                                    IndexBinaryOp.ADD,
                                    linear,
                                    exprGraph.indexBinary(
                                            IndexBinaryOp.ADD,
                                            exprGraph.indexBinary(
                                                    IndexBinaryOp.MULTIPLY,
                                                    exprGraph.indexConst(key0),
                                                    exprGraph.indexConst(1009)),
                                            exprGraph.indexBinary(
                                                    IndexBinaryOp.MULTIPLY,
                                                    exprGraph.indexConst(key1),
                                                    exprGraph.indexConst(9176))));
                    // Keep LCG state bounded to prevent overflow in nextState multiplication.
                    // Use 2^48 as the modulus which is large enough to maintain good randomness
                    // but small enough to prevent 64-bit overflow (max intermediate: 2^48 * 1664525
                    // < 2^63). Use bitwise AND with (2^48 - 1) for efficient power-of-2 modulus.
                    LIRExprNode state0 = modPowerOfTwo(nextState(seed), 281474976710655L);

                    if (node.dataType() == DataType.FP32) {
                        LIRExprNode u24 = modPositive(state0, 16777216L);
                        LIRExprNode numerator = exprGraph.scalarFromIndex(u24, DataType.FP32);
                        LIRExprNode denom =
                                exprGraph.scalarConst(
                                        Float.floatToRawIntBits(16777216.0f), DataType.FP32);
                        return exprGraph.scalarBinary(BinaryOperator.DIVIDE, numerator, denom);
                    }

                    LIRExprNode state1 = modPowerOfTwo(nextState(state0), 281474976710655L);
                    LIRExprNode hi26 = modPositive(state0, 67108864L);
                    LIRExprNode lo27 = modPositive(state1, 134217728L);
                    LIRExprNode bits53 =
                            exprGraph.indexBinary(
                                    IndexBinaryOp.ADD,
                                    exprGraph.indexBinary(
                                            IndexBinaryOp.MULTIPLY,
                                            hi26,
                                            exprGraph.indexConst(134217728)),
                                    lo27);
                    LIRExprNode numerator = exprGraph.scalarFromIndex(bits53, DataType.FP64);
                    LIRExprNode denom =
                            exprGraph.scalarConst(
                                    Double.doubleToRawLongBits(9007199254740992.0), DataType.FP64);
                    return exprGraph.scalarBinary(BinaryOperator.DIVIDE, numerator, denom);
                });
    }

    private LIRExprNode nextState(LIRExprNode value) {
        return exprGraph.indexBinary(
                IndexBinaryOp.ADD,
                exprGraph.indexBinary(IndexBinaryOp.MULTIPLY, value, exprGraph.indexConst(1664525)),
                exprGraph.indexConst(1013904223));
    }

    private LIRExprNode modPositive(LIRExprNode value, long bound) {
        LIRExprNode mod =
                exprGraph.indexBinary(IndexBinaryOp.MODULO, value, exprGraph.indexConst(bound));
        return exprGraph.indexBinary(
                IndexBinaryOp.MODULO,
                exprGraph.indexBinary(IndexBinaryOp.ADD, mod, exprGraph.indexConst(bound)),
                exprGraph.indexConst(bound));
    }

    /**
     * Computes value mod 2^n using bitwise AND with (2^n - 1). This is much faster than regular
     * modulo on GPUs. The mask parameter should be (2^n - 1), e.g., for 2^48, use 281474976710655L.
     */
    private LIRExprNode modPowerOfTwo(LIRExprNode value, long mask) {
        return exprGraph.indexBinary(IndexBinaryOp.BITWISE_AND, value, exprGraph.indexConst(mask));
    }
}
