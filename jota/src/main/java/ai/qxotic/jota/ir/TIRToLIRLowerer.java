package ai.qxotic.jota.ir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.tir.*;
import ai.qxotic.jota.ir.lir.v2.LIRV2WorklistPass;
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
public class TIRToLIRLowerer implements TIRVisitor<ScalarExpr> {

    private final Map<TIRNode, BufferRef> inputBuffers = new IdentityHashMap<>();
    private final Map<TIRNode, ai.qxotic.jota.ir.lir.ScalarInput> inputScalars =
            new IdentityHashMap<>();
    private final Map<TIRNode, BufferRef> outputBuffers = new IdentityHashMap<>();
    private final List<LIRInput> allInputs = new ArrayList<>();
    private final List<BufferRef> allOutputs = new ArrayList<>();
    private final IdentityHashMap<TIRNode, ScalarRef> scalarRefCache = new IdentityHashMap<>();
    private final List<ScalarLet> scalarLets = new ArrayList<>();
    private int nextId = 0;
    private int nextScalarId = 0;
    private boolean scalarCachingEnabled = false;

    // Current loop indices for the output iteration
    private final List<IndexVar> loopIndices = new ArrayList<>();

    /** Lowers a TIRGraph to an LIRGraph. */
    public LIRGraph lower(TIRGraph tirGraph) {
        inputBuffers.clear();
        inputScalars.clear();
        outputBuffers.clear();
        allInputs.clear();
        allOutputs.clear();
        loopIndices.clear();
        scalarRefCache.clear();
        scalarLets.clear();
        scalarCachingEnabled = false;
        nextId = 0;
        nextScalarId = 0;

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

            if (input instanceof ai.qxotic.jota.ir.tir.ScalarInput scalarInput) {
                ai.qxotic.jota.ir.lir.ScalarInput lirInput =
                        new ai.qxotic.jota.ir.lir.ScalarInput(nextId++, scalarInput.dataType());
                inputScalars.put(input, lirInput);
                allInputs.add(lirInput);
                continue;
            }

            // Check if this is a ScalarConstant or a ViewTransform chain wrapping one
            ScalarConstant underlyingScalar = extractUnderlyingScalarConstant(input);
            if (underlyingScalar != null) {
                // Scalar inputs become scalar parameters (passed by value)
                ai.qxotic.jota.ir.lir.ScalarInput scalarInput =
                        new ai.qxotic.jota.ir.lir.ScalarInput(
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
        List<LIRNode> statements = new ArrayList<>();
        for (int i = 0; i < tirGraph.outputs().size(); i++) {
            TIRNode output = tirGraph.outputs().get(i);
            BufferRef outBuf = allOutputs.get(i);
            LIRNode stmt = lowerOutput(output, outBuf);
            statements.add(stmt);
        }

        LIRNode body = statements.size() == 1 ? statements.getFirst() : new Block(statements);

        LIRGraph graph = new LIRGraph(allInputs, allOutputs, body);
        return new LIRV2WorklistPass().run(graph);
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
        return node instanceof IotaConstant;
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
        return BufferRef.of(nextId++, node.dataType(), layoutForNode(node));
    }

    private Layout layoutForNode(TIRNode node) {
        if (node instanceof TensorInput input) {
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
    private LIRNode lowerOutput(TIRNode output, BufferRef outBuf) {
        // Pure reductions (output is directly a ReductionOp) are handled here
        if (output instanceof ReductionOp reduction) {
            return lowerReduction(reduction, outBuf, null);
        }

        ReductionOp reductionRoot = findReductionRoot(output);
        if (reductionRoot != null) {
            ScalarExpr postOpValue =
                    buildPostOpExpr(
                            output, reductionRoot, new ScalarRef("acc0", reductionRoot.dataType()));
            return lowerReduction(reductionRoot, outBuf, postOpValue);
        }

        Layout outLayout = layoutForNode(output).flatten();
        int rank = outLayout.shape().flatRank();

        // Create loop index variables
        loopIndices.clear();
        for (int i = 0; i < rank; i++) {
            loopIndices.add(new IndexVar("i" + i));
        }

        // Compute the scalar expression for this output
        beginScalarCaching();
        ScalarExpr value = output.accept(this);
        List<ScalarLet> localLets = new ArrayList<>(scalarLets);
        endScalarCaching();

        // Compute output offset
        IndexExpr outOffset = computeOffset(loopIndices, outBuf.byteStrides());

        // Create the store
        Store store = new Store(outBuf, outOffset, value);

        LIRNode storeNode = store;
        if (!localLets.isEmpty()) {
            List<LIRNode> statements = new ArrayList<>(localLets.size() + 1);
            statements.addAll(localLets);
            statements.add(store);
            storeNode = new Block(statements);
        }

        // Wrap in nested loops (innermost first)
        LIRNode body = ensureYield(storeNode);
        for (int i = rank - 1; i >= 0; i--) {
            long bound = outLayout.shape().flatAt(i);
            body =
                    new StructuredFor(
                            loopIndices.get(i).name(),
                            new IndexConst(0),
                            new IndexConst(bound),
                            new IndexConst(1),
                            List.of(),
                            ensureYield(body));
        }

        return body;
    }

    private void beginScalarCaching() {
        scalarRefCache.clear();
        scalarLets.clear();
        nextScalarId = 0;
        scalarCachingEnabled = true;
    }

    private void endScalarCaching() {
        scalarCachingEnabled = false;
    }

    private ScalarExpr cacheScalar(TIRNode node, Supplier<ScalarExpr> builder) {
        if (!scalarCachingEnabled) {
            return builder.get();
        }
        ScalarRef cached = scalarRefCache.get(node);
        if (cached != null) {
            return cached;
        }
        ScalarExpr value = builder.get();
        String name = "t" + nextScalarId++;
        ScalarRef ref = new ScalarRef(name, value.dataType());
        scalarLets.add(new ScalarLet(name, value));
        scalarRefCache.put(node, ref);
        return ref;
    }

    /** Lowers a reduction operation to LIR with accumulators. */
    private LIRNode lowerReduction(ReductionOp reduction, BufferRef outBuf) {
        return lowerReduction(reduction, outBuf, null);
    }

    private LIRNode lowerReduction(
            ReductionOp reduction, BufferRef outBuf, ScalarExpr postOpValue) {
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
        List<IndexVar> outerIndices = new ArrayList<>();
        List<IndexVar> innerIndices = new ArrayList<>();
        List<Long> outerBounds = new ArrayList<>();
        List<Long> innerBounds = new ArrayList<>();

        for (int i = 0; i < inputRank; i++) {
            IndexVar idx = new IndexVar("i" + i);
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
        ScalarExpr inputValue = input.accept(this);

        LIRNode reductionBody =
                buildReductionLoop(innerIndices, innerBounds, dtype, op, inputValue);

        // Read accumulator and store to output (apply post-ops if provided)
        ScalarExpr read = new ScalarRef("acc0", dtype);
        ScalarExpr storeValue = (postOpValue != null) ? postOpValue : read;

        // Compute output offset from outer indices only
        IndexExpr outOffset;
        if (outerIndices.isEmpty()) {
            outOffset = new IndexConst(0);
        } else {
            outOffset = computeOffset(outerIndices, outBuf.byteStrides());
        }
        Store store = new Store(outBuf, outOffset, storeValue);

        // Combine: declare acc, reduction loops, store result
        Block innerBlock = new Block(List.of(reductionBody, store));

        // Wrap in outer loops (over non-reduced dimensions)
        LIRNode body = innerBlock;
        for (int i = outerIndices.size() - 1; i >= 0; i--) {
            body =
                    new StructuredFor(
                            outerIndices.get(i).name(),
                            new IndexConst(0),
                            new IndexConst(outerBounds.get(i)),
                            new IndexConst(1),
                            List.of(),
                            ensureYield(body));
        }

        return body;
    }

    private ReductionOp findReductionRoot(TIRNode node) {
        ReductionOp[] found = new ReductionOp[1];
        findReductionRootRec(node, found, new java.util.IdentityHashMap<>());
        return found[0];
    }

    private void findReductionRootRec(
            TIRNode node,
            ReductionOp[] found,
            java.util.IdentityHashMap<TIRNode, Boolean> visited) {
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
            case ViewTransform vt -> findReductionRootRec(vt.input(), found, visited);
            case Contiguous contig -> findReductionRootRec(contig.input(), found, visited);
            default -> {}
        }
    }

    private ScalarExpr buildPostOpExpr(
            TIRNode node, ReductionOp reduction, ScalarExpr reductionValue) {
        if (node == reduction) {
            return reductionValue;
        }
        switch (node) {
            case UnaryOp op -> {
                ScalarExpr input = buildPostOpExpr(op.input(), reduction, reductionValue);
                return new ScalarUnary(op.op(), input);
            }
            case BinaryOp op -> {
                ScalarExpr left = buildPostOpExpr(op.left(), reduction, reductionValue);
                ScalarExpr right = buildPostOpExpr(op.right(), reduction, reductionValue);
                return new ScalarBinary(op.op(), left, right);
            }
            case TernaryOp op -> {
                ScalarExpr cond = buildPostOpExpr(op.cond(), reduction, reductionValue);
                ScalarExpr trueVal = buildPostOpExpr(op.trueExpr(), reduction, reductionValue);
                ScalarExpr falseVal = buildPostOpExpr(op.falseExpr(), reduction, reductionValue);
                return new ScalarTernary(cond, trueVal, falseVal);
            }
            case CastOp op -> {
                ScalarExpr input = buildPostOpExpr(op.input(), reduction, reductionValue);
                return new ScalarCast(input, op.targetDataType());
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
            case TensorInput __ ->
                    throw new UnsupportedOperationException(
                            "Post-ops after reductions must be element-wise on the reduction result");
            case ai.qxotic.jota.ir.tir.ScalarInput si -> {
                ScalarExpr scalar = inputScalars.get(si);
                if (scalar == null) {
                    throw new IllegalStateException("Unknown ScalarInput: " + si.id());
                }
                return scalar;
            }
            case IotaConstant __ ->
                    throw new UnsupportedOperationException(
                            "Post-ops after reductions must be element-wise on the reduction result");
        }
    }

    private LIRNode buildReductionLoop(
            List<IndexVar> innerIndices,
            List<Long> innerBounds,
            DataType dtype,
            ReductionOperator op,
            ScalarExpr inputValue) {
        int depth = innerIndices.size();
        if (depth == 0) {
            ScalarExpr combined = combineAccumulator(new ScalarRef("acc0", dtype), inputValue, op);
            return new Yield(List.of(combined));
        }

        String[] accNames = new String[depth];
        for (int i = 0; i < depth; i++) {
            accNames[i] = "acc" + i;
        }

        ScalarExpr identity = createIdentityLiteral(dtype, op);

        ScalarExpr combined =
                combineAccumulator(new ScalarRef(accNames[depth - 1], dtype), inputValue, op);
        LIRNode inner = new Yield(List.of(combined));

        for (int level = depth - 1; level >= 0; level--) {
            String accName = accNames[level];
            ScalarExpr init = (level == 0) ? identity : new ScalarRef(accNames[level - 1], dtype);

            LIRNode body;
            if (level == depth - 1) {
                body = inner;
            } else {
                body =
                        new Block(
                                List.of(
                                        inner,
                                        new Yield(
                                                List.of(
                                                        new ScalarRef(
                                                                accNames[level + 1], dtype)))));
            }

            inner =
                    new StructuredFor(
                            innerIndices.get(level).name(),
                            new IndexConst(0),
                            new IndexConst(innerBounds.get(level)),
                            new IndexConst(1),
                            List.of(new LoopIterArg(accName, dtype, init)),
                            body);
        }

        return inner;
    }

    private ScalarExpr combineAccumulator(ScalarExpr acc, ScalarExpr value, ReductionOperator op) {
        // Cast value to accumulator type if needed
        ScalarExpr castValue = value;
        if (value.dataType() != acc.dataType()) {
            castValue = new ScalarCast(value, acc.dataType());
        }
        return switch (op) {
            case SUM -> new ScalarBinary(BinaryOperator.ADD, acc, castValue);
            case PROD -> new ScalarBinary(BinaryOperator.MULTIPLY, acc, castValue);
            case MIN -> new ScalarBinary(BinaryOperator.MIN, acc, castValue);
            case MAX -> new ScalarBinary(BinaryOperator.MAX, acc, castValue);
        };
    }

    private ScalarExpr createIdentityLiteral(DataType dtype, ReductionOperator op) {
        return switch (op) {
            case SUM -> ScalarLiteral.ofRawBits(identityZeroBits(dtype), dtype);
            case PROD -> ScalarLiteral.ofRawBits(identityOneBits(dtype), dtype);
            case MIN -> ScalarLiteral.ofRawBits(identityMinBits(dtype), dtype);
            case MAX -> ScalarLiteral.ofRawBits(identityMaxBits(dtype), dtype);
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

    private LIRNode ensureYield(LIRNode body) {
        if (body instanceof Yield) {
            return body;
        }
        if (body instanceof Block block) {
            if (!block.statements().isEmpty() && block.statements().getLast() instanceof Yield) {
                return body;
            }
        }
        return new Block(List.of(body, Yield.empty()));
    }

    /** Computes byte offset from indices and byte strides. */
    private IndexExpr computeOffset(List<? extends IndexExpr> indices, long[] strides) {
        if (indices.isEmpty()) {
            return new IndexConst(0);
        }

        IndexExpr offset = null;
        for (int i = 0; i < indices.size(); i++) {
            if (strides[i] == 0) {
                continue; // Broadcasting dimension - skip
            }
            IndexExpr term = IndexBinary.multiply(indices.get(i), new IndexConst(strides[i]));
            offset = (offset == null) ? term : IndexBinary.add(offset, term);
        }
        return offset != null ? offset : new IndexConst(0);
    }

    // ==================== TIR Visitor Methods ====================

    @Override
    public ScalarExpr visitTensorInput(TensorInput node) {
        return cacheScalar(
                node,
                () -> {
                    BufferRef buf = inputBuffers.get(node);
                    if (buf == null) {
                        throw new IllegalStateException("Unknown TensorInput: " + node.id());
                    }
                    IndexExpr offset = computeOffset(loopIndices, buf.byteStrides());
                    return new ScalarLoad(buf, offset);
                });
    }

    @Override
    public ScalarExpr visitScalarInput(ai.qxotic.jota.ir.tir.ScalarInput node) {
        ScalarExpr scalar = inputScalars.get(node);
        if (scalar == null) {
            throw new IllegalStateException("Unknown ScalarInput: " + node.id());
        }
        return scalar;
    }

    @Override
    public ScalarExpr visitScalarConstant(ScalarConstant node) {
        // Check if this scalar is an explicit input (dynamic parameter)
        ai.qxotic.jota.ir.lir.ScalarInput scalarInput = inputScalars.get(node);
        if (scalarInput != null) {
            // Scalar input - reference the scalar parameter directly (passed by value)
            return scalarInput;
        }
        // Not an input - inline as literal
        return new ScalarLiteral(node.rawBits(), node.dataType());
    }

    @Override
    public ScalarExpr visitUnaryOp(UnaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    ScalarExpr input = node.input().accept(this);
                    return new ScalarUnary(node.op(), input);
                });
    }

    @Override
    public ScalarExpr visitBinaryOp(BinaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    ScalarExpr left = node.left().accept(this);
                    ScalarExpr right = node.right().accept(this);
                    return new ScalarBinary(node.op(), left, right);
                });
    }

    @Override
    public ScalarExpr visitTernaryOp(TernaryOp node) {
        return cacheScalar(
                node,
                () -> {
                    ScalarExpr cond = node.cond().accept(this);
                    ScalarExpr trueVal = node.trueExpr().accept(this);
                    ScalarExpr falseVal = node.falseExpr().accept(this);
                    return new ScalarTernary(cond, trueVal, falseVal);
                });
    }

    @Override
    public ScalarExpr visitCastOp(CastOp node) {
        return cacheScalar(
                node,
                () -> {
                    ScalarExpr input = node.input().accept(this);
                    return new ScalarCast(input, node.targetDataType());
                });
    }

    @Override
    public ScalarExpr visitReductionOp(ReductionOp node) {
        // Reductions should have been handled in lowerOutput, not visited directly
        // This can happen if a reduction is used in a context that requires scalar evaluation
        // In such cases, we need to handle it - for now, throw a more helpful error
        throw new UnsupportedOperationException(
                "ReductionOp lowering not yet implemented - requires separate handling. "
                        + "Reductions with post-operations (e.g., .add(), .cast()) are not yet supported.");
    }

    @Override
    public ScalarExpr visitViewTransform(ViewTransform node) {
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

                    ScalarExpr result;
                    if (baseNode instanceof TensorInput tensorInput) {
                        result = lowerTensorInputWithViewChain(tensorInput, chain, node);
                    } else if (baseNode instanceof ai.qxotic.jota.ir.tir.ScalarInput scalarInput) {
                        ScalarExpr scalar = inputScalars.get(scalarInput);
                        if (scalar == null) {
                            throw new IllegalStateException("Unknown ScalarInput: " + scalarInput.id());
                        }
                        result = scalar;
                    } else if (baseNode instanceof IotaConstant iotaConstant) {
                        result = lowerIotaConstantWithViewChain(iotaConstant, chain, node);
                    } else {
                        // For other input types, delegate normally
                        return node.input().accept(this);
                    }

                    // Apply cast if needed
                    if (castType != null && castType != result.dataType()) {
                        return new ScalarCast(result, castType);
                    }
                    return result;
                });
    }

    private ScalarExpr lowerTensorInputWithViewChain(
            TensorInput tensorInput, List<ViewTransform> chain, ViewTransform node) {
        BufferRef buf = inputBuffers.get(tensorInput);
        if (buf == null) {
            throw new IllegalStateException("Unknown TensorInput: " + tensorInput.id());
        }

        // Check if any transform in the chain needs lazy indexing
        boolean needsLazy = chain.stream().anyMatch(ViewTransform::needsLazyIndexing);

        if (!needsLazy) {
            // Simple case: use the final view's strides directly
            Layout viewLayout = node.layout().flatten();
            long[] viewByteStrides = toByteStrides(viewLayout, node.dataType());
            IndexExpr offset = computeOffset(loopIndices, viewByteStrides);
            return new ScalarLoad(buf, offset);
        }

        // Lazy indexing: compose index expressions by walking the chain in reverse
        // Start with output loop indices and transform back to input coordinates
        List<IndexExpr> indices = new ArrayList<>(loopIndices);

        // Apply transforms in REVERSE order (from output to input)
        for (int i = chain.size() - 1; i >= 0; i--) {
            ViewTransform vt = chain.get(i);
            indices = applyInverseTransform(vt, indices);
        }

        // Compute final offset using the BASE input's strides
        IndexExpr offset = computeOffset(indices, buf.byteStrides());
        return new ScalarLoad(buf, offset);
    }

    private ScalarExpr lowerIotaConstantWithViewChain(
            IotaConstant iotaConstant, List<ViewTransform> chain, ViewTransform node) {
        // For IotaConstant, we compute the linear index from the inverse-transformed coordinates
        // Start with output loop indices and transform back to IotaConstant's original space
        List<IndexExpr> indices = new ArrayList<>(loopIndices);

        // Apply transforms in REVERSE order (from output to input)
        for (int i = chain.size() - 1; i >= 0; i--) {
            ViewTransform vt = chain.get(i);
            indices = applyInverseTransform(vt, indices);
        }

        // Now 'indices' are in IotaConstant's original flat space
        // Compute linear index from these coordinates using IotaConstant's layout
        Layout iotaLayout = Layout.rowMajor(iotaConstant.shape()).flatten();
        IndexExpr linearIdx = null;
        long stride = 1;
        // Compute row-major linear index: sum(idx[i] * stride[i])
        for (int i = indices.size() - 1; i >= 0; i--) {
            IndexExpr term = IndexBinary.multiply(indices.get(i), new IndexConst(stride));
            linearIdx = (linearIdx == null) ? term : IndexBinary.add(term, linearIdx);
            stride *= (i < iotaLayout.shape().flatRank()) ? iotaLayout.shape().flatAt(i) : 1;
        }

        if (linearIdx == null) {
            linearIdx = new IndexConst(0);
        }

        // Cast index to the output type
        DataType targetType = iotaConstant.dataType();
        ScalarExpr indexValue = new ScalarFromIndex(linearIdx);
        if (targetType == DataType.I64) {
            return indexValue;
        } else {
            return new ScalarCast(indexValue, targetType);
        }
    }

    /**
     * Applies the inverse of a view transformation to convert output indices to input indices.
     *
     * <p>For example, if the transform is Transpose([1, 0]) on shape (A, B) -> (B, A), the inverse
     * maps (i, j) in output space to (j, i) in input space.
     */
    private List<IndexExpr> applyInverseTransform(ViewTransform vt, List<IndexExpr> outputIndices) {
        return switch (vt.kind()) {
            case ViewKind.Transpose transpose -> {
                // Transpose: permute indices using inverse permutation
                int[] invPerm = transpose.inverse();
                List<IndexExpr> result = new ArrayList<>(outputIndices.size());
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
                IndexExpr linearIdx = flattenIndices(outputIndices, toShape);
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

                List<IndexExpr> result = new ArrayList<>();
                for (int i = 0; i < fromRank; i++) {
                    long fromDim = fromShape.flatAt(i);
                    if (fromDim == 1) {
                        // Broadcast dimension - input index is always 0
                        result.add(new IndexConst(0));
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
                List<IndexExpr> result = new ArrayList<>();
                for (int i = 0; i < fromShape.flatRank(); i++) {
                    long fromDim = fromShape.flatAt(i);
                    if (fromDim == 1) {
                        result.add(new IndexConst(0));
                    } else {
                        result.add(outputIndices.get(i));
                    }
                }
                yield result;
            }
            case ViewKind.Slice slice -> {
                // Slice: output index maps to input index with offset and step
                // input_idx = start + output_idx * step
                List<IndexExpr> result = new ArrayList<>(outputIndices);
                IndexExpr outIdx = outputIndices.get(slice.axis());
                IndexExpr inIdx =
                        IndexBinary.add(
                                new IndexConst(slice.start()),
                                IndexBinary.multiply(outIdx, new IndexConst(slice.step())));
                result.set(slice.axis(), inIdx);
                yield result;
            }
        };
    }

    /** Flattens multi-dimensional indices to a linear index using row-major order. */
    private IndexExpr flattenIndices(List<IndexExpr> indices, Shape shape) {
        if (indices.isEmpty()) {
            return new IndexConst(0);
        }
        IndexExpr linear = null;
        long stride = 1;
        for (int i = (int) shape.flatRank() - 1; i >= 0; i--) {
            IndexExpr term = IndexBinary.multiply(indices.get(i), new IndexConst(stride));
            linear = (linear == null) ? term : IndexBinary.add(term, linear);
            stride *= shape.flatAt(i);
        }
        return linear != null ? linear : new IndexConst(0);
    }

    /** Decomposes a linear index into multi-dimensional indices using row-major order. */
    private List<IndexExpr> unflattenIndex(IndexExpr linearIdx, Shape shape) {
        int rank = (int) shape.flatRank();
        if (rank == 0) {
            return List.of();
        }
        List<IndexExpr> indices = new ArrayList<>(rank);
        // Compute strides (row-major)
        long[] strides = new long[rank];
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape.flatAt(i + 1);
        }
        // Decompose: idx[i] = (linear / stride[i]) % dim[i]
        for (int i = 0; i < rank; i++) {
            IndexExpr divided = IndexBinary.divide(linearIdx, new IndexConst(strides[i]));
            IndexExpr idx = IndexBinary.modulo(divided, new IndexConst(shape.flatAt(i)));
            indices.add(idx);
        }
        return indices;
    }

    @Override
    public ScalarExpr visitContiguous(Contiguous node) {
        // Contiguous forces a copy, but at this level we just read the input
        return cacheScalar(node, () -> node.input().accept(this));
    }

    @Override
    public ScalarExpr visitIotaConstant(IotaConstant node) {
        // Iota returns the index value - need to compute the linear index
        return cacheScalar(
                node,
                () -> {
                    if (loopIndices.isEmpty()) {
                        return ScalarLiteral.ofLong(0);
                    }

                    // Compute linear index from loop indices
                    Layout layout = Layout.rowMajor(node.shape()).flatten();
                    IndexExpr linearIdx = null;
                    long stride = 1;
                    for (int i = layout.shape().flatRank() - 1; i >= 0; i--) {
                        IndexExpr term =
                                IndexBinary.multiply(loopIndices.get(i), new IndexConst(stride));
                        linearIdx = (linearIdx == null) ? term : IndexBinary.add(term, linearIdx);
                        stride *= layout.shape().flatAt(i);
                    }

                    // Cast index to the output type
                    if (linearIdx == null) {
                        linearIdx = new IndexConst(0);
                    }

                    // IndexExpr is I64, cast to target type if needed
                    DataType targetType = node.dataType();
                    ScalarExpr indexValue = new ScalarFromIndex(linearIdx);
                    if (targetType == DataType.I64) {
                        return indexValue;
                    }
                    return new ScalarCast(indexValue, targetType);
                });
    }
}
