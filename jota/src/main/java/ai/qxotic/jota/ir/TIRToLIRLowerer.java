package ai.qxotic.jota.ir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.tir.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lowers TIR (Tensor IR) graphs to LIR (Loop-level IR) graphs.
 *
 * <p>TIR represents tensor operations semantically, while LIR makes loops and memory access
 * explicit.
 */
public class TIRToLIRLowerer implements TIRVisitor<ScalarExpr> {

    private final Map<TIRNode, BufferRef> inputBuffers = new HashMap<>();
    private final Map<TIRNode, ScalarInput> inputScalars = new HashMap<>();
    private final Map<TIRNode, BufferRef> outputBuffers = new HashMap<>();
    private final List<LIRInput> allInputs = new ArrayList<>();
    private final List<BufferRef> allOutputs = new ArrayList<>();
    private int nextId = 0;

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
        nextId = 0;

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

            // Check if this is a ScalarConstant or a ViewTransform chain wrapping one
            ScalarConstant underlyingScalar = extractUnderlyingScalarConstant(input);
            if (underlyingScalar != null) {
                // Scalar inputs become scalar parameters (passed by value)
                ScalarInput scalarInput = new ScalarInput(nextId++, underlyingScalar.dataType());
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

        return new LIRGraph(allInputs, allOutputs, body);
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
     * Extracts the underlying ScalarConstant from a node if it's a ScalarConstant
     * or a chain of ViewTransforms wrapping a ScalarConstant.
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
        return BufferRef.of(nextId++, node.dataType(), node.layout());
    }

    private BufferRef createOutputBufferRef(TIRNode node) {
        // Outputs are always contiguous row-major
        return BufferRef.of(
                nextId++, node.dataType(), Layout.rowMajor(node.layout().shape()));
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
        // Handle reductions specially
        if (output instanceof ReductionOp reduction) {
            return lowerReduction(reduction, outBuf);
        }

        Layout outLayout = output.layout().flatten();
        int rank = outLayout.shape().flatRank();

        // Create loop index variables
        loopIndices.clear();
        for (int i = 0; i < rank; i++) {
            loopIndices.add(new IndexVar("i" + i));
        }

        // Compute the scalar expression for this output
        ScalarExpr value = output.accept(this);

        // Compute output offset
        IndexExpr outOffset = computeOffset(loopIndices, outBuf.byteStrides());

        // Create the store
        Store store = new Store(outBuf, outOffset, value);

        // Wrap in nested loops (innermost first)
        LIRNode body = store;
        for (int i = rank - 1; i >= 0; i--) {
            long bound = outLayout.shape().flatAt(i);
            body = Loop.parallel(loopIndices.get(i).name(), bound, body);
        }

        return body;
    }

    /** Lowers a reduction operation to LIR with accumulators. */
    private LIRNode lowerReduction(ReductionOp reduction, BufferRef outBuf) {
        TIRNode input = reduction.input();
        Layout inputLayout = input.layout().flatten();
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

        // Create accumulator
        String accName = "acc";
        Accumulator acc = createAccumulator(accName, dtype, op);

        // Load value from input and update accumulator
        ScalarExpr inputValue = input.accept(this);
        AccumulatorUpdate update = new AccumulatorUpdate(accName, inputValue);

        // Build the reduction loop (inner loops over reduced axes)
        LIRNode reductionBody = update;
        for (int i = innerIndices.size() - 1; i >= 0; i--) {
            reductionBody =
                    Loop.sequential(innerIndices.get(i).name(), innerBounds.get(i), reductionBody);
        }

        // Read accumulator and store to output
        AccumulatorRead read = new AccumulatorRead(accName, dtype);

        // Compute output offset from outer indices only
        IndexExpr outOffset;
        if (outerIndices.isEmpty()) {
            outOffset = new IndexConst(0);
        } else {
            outOffset = computeOffset(outerIndices, outBuf.byteStrides());
        }
        Store store = new Store(outBuf, outOffset, read);

        // Combine: declare acc, reduction loops, store result
        Block innerBlock = new Block(List.of(acc, reductionBody, store));

        // Wrap in outer loops (over non-reduced dimensions)
        LIRNode body = innerBlock;
        for (int i = outerIndices.size() - 1; i >= 0; i--) {
            body = Loop.parallel(outerIndices.get(i).name(), outerBounds.get(i), body);
        }

        return body;
    }

    private Accumulator createAccumulator(String name, DataType dtype, ReductionOperator op) {
        return switch (op) {
            case SUM -> Accumulator.sum(name, dtype);
            case PROD -> Accumulator.product(name, dtype);
            case MIN -> Accumulator.min(name, dtype);
            case MAX -> Accumulator.max(name, dtype);
        };
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
        BufferRef buf = inputBuffers.get(node);
        if (buf == null) {
            throw new IllegalStateException("Unknown TensorInput: " + node.id());
        }
        IndexExpr offset = computeOffset(loopIndices, buf.byteStrides());
        return new ScalarLoad(buf, offset);
    }

    @Override
    public ScalarExpr visitScalarConstant(ScalarConstant node) {
        // Check if this scalar is an explicit input (dynamic parameter)
        ScalarInput scalarInput = inputScalars.get(node);
        if (scalarInput != null) {
            // Scalar input - reference the scalar parameter directly (passed by value)
            return scalarInput;
        }
        // Not an input - inline as literal
        return new ScalarLiteral(node.rawBits(), node.dataType());
    }

    @Override
    public ScalarExpr visitUnaryOp(UnaryOp node) {
        ScalarExpr input = node.input().accept(this);
        return new ScalarUnary(node.op(), input);
    }

    @Override
    public ScalarExpr visitBinaryOp(BinaryOp node) {
        ScalarExpr left = node.left().accept(this);
        ScalarExpr right = node.right().accept(this);
        return new ScalarBinary(node.op(), left, right);
    }

    @Override
    public ScalarExpr visitTernaryOp(TernaryOp node) {
        ScalarExpr cond = node.cond().accept(this);
        ScalarExpr trueVal = node.trueExpr().accept(this);
        ScalarExpr falseVal = node.falseExpr().accept(this);
        return new ScalarTernary(cond, trueVal, falseVal);
    }

    @Override
    public ScalarExpr visitCastOp(CastOp node) {
        ScalarExpr input = node.input().accept(this);
        return new ScalarCast(input, node.targetDataType());
    }

    @Override
    public ScalarExpr visitReductionOp(ReductionOp node) {
        // Reductions are complex - they need accumulators
        // For now, throw unsupported
        throw new UnsupportedOperationException(
                "ReductionOp lowering not yet implemented - requires separate handling");
    }

    @Override
    public ScalarExpr visitViewTransform(ViewTransform node) {
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
        Layout iotaLayout = iotaConstant.layout().flatten();
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
        return node.input().accept(this);
    }

    @Override
    public ScalarExpr visitIotaConstant(IotaConstant node) {
        // Iota returns the index value - need to compute the linear index
        if (loopIndices.isEmpty()) {
            return ScalarLiteral.ofLong(0);
        }

        // Compute linear index from loop indices
        Layout layout = node.layout().flatten();
        IndexExpr linearIdx = null;
        long stride = 1;
        for (int i = layout.shape().flatRank() - 1; i >= 0; i--) {
            IndexExpr term = IndexBinary.multiply(loopIndices.get(i), new IndexConst(stride));
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
        } else {
            return new ScalarCast(indexValue, targetType);
        }
    }
}
