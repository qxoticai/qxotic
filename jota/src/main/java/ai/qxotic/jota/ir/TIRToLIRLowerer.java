package ai.qxotic.jota.ir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
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
    private final Map<TIRNode, BufferRef> outputBuffers = new HashMap<>();
    private final List<BufferRef> allInputs = new ArrayList<>();
    private final List<BufferRef> allOutputs = new ArrayList<>();
    private int nextBufferId = 0;

    // Current loop indices for the output iteration
    private final List<IndexVar> loopIndices = new ArrayList<>();

    /** Lowers a TIRGraph to an LIRGraph. */
    public LIRGraph lower(TIRGraph tirGraph) {
        inputBuffers.clear();
        outputBuffers.clear();
        allInputs.clear();
        allOutputs.clear();
        loopIndices.clear();
        nextBufferId = 0;

        // Create input buffers
        for (TIRNode input : tirGraph.inputs()) {
            BufferRef buf = createBufferRef(input);
            inputBuffers.put(input, buf);
            allInputs.add(buf);
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

    private BufferRef createBufferRef(TIRNode node) {
        Layout layout = node.layout().flatten();
        long[] shape = toArray(layout.shape());
        long[] strides = toByteStrides(layout, node.dataType());
        return new BufferRef(nextBufferId++, node.dataType(), shape, strides);
    }

    private BufferRef createOutputBufferRef(TIRNode node) {
        Layout layout = node.layout().flatten();
        long[] shape = toArray(layout.shape());
        // Outputs are always contiguous row-major
        return BufferRef.contiguous(nextBufferId++, node.dataType(), shape);
    }

    private long[] toArray(ai.qxotic.jota.Shape shape) {
        long[] arr = new long[shape.flatRank()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = shape.flatAt(i);
        }
        return arr;
    }

    private long[] toByteStrides(Layout layout, DataType dtype) {
        int rank = layout.shape().flatRank();
        long[] strides = new long[rank];
        long byteSize = dtype.byteSize();
        for (int i = 0; i < rank; i++) {
            strides[i] = layout.stride().flatAt(i) * byteSize;
        }
        return strides;
    }

    /** Lowers a single output expression to LIR loop nest. */
    private LIRNode lowerOutput(TIRNode output, BufferRef outBuf) {
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
        IndexExpr outOffset = computeOffset(loopIndices, outBuf.strides());

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

    /** Computes byte offset from indices and byte strides. */
    private IndexExpr computeOffset(List<IndexVar> indices, long[] strides) {
        if (indices.isEmpty()) {
            return new IndexConst(0);
        }

        IndexExpr offset = null;
        for (int i = 0; i < indices.size(); i++) {
            if (strides[i] == 0) {
                continue; // Broadcasting dimension - skip
            }
            IndexExpr term = IndexBinary.mul(indices.get(i), new IndexConst(strides[i]));
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
        IndexExpr offset = computeOffset(loopIndices, buf.strides());
        return new ScalarLoad(buf, offset);
    }

    @Override
    public ScalarExpr visitScalarConstant(ScalarConstant node) {
        return new ScalarConst(node.rawBits(), node.dataType());
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
        // ViewTransform changes layout but not data - just lower the input with new strides
        return node.input().accept(this);
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
            return ScalarConst.ofLong(0);
        }

        // Compute linear index from loop indices
        Layout layout = node.layout().flatten();
        IndexExpr linearIdx = null;
        long stride = 1;
        for (int i = layout.shape().flatRank() - 1; i >= 0; i--) {
            IndexExpr term = IndexBinary.mul(loopIndices.get(i), new IndexConst(stride));
            linearIdx = (linearIdx == null) ? term : IndexBinary.add(term, linearIdx);
            stride *= layout.shape().flatAt(i);
        }

        // Cast index to the output type
        if (linearIdx == null) {
            linearIdx = new IndexConst(0);
        }

        // IndexExpr is I64, cast to target type if needed
        DataType targetType = node.dataType();
        if (targetType == DataType.I64) {
            // Can't directly use IndexExpr as ScalarExpr - need to convert
            // For now, return as constant 0 placeholder
            throw new UnsupportedOperationException(
                    "IotaConstant lowering requires IndexExpr to ScalarExpr conversion");
        }

        throw new UnsupportedOperationException("IotaConstant lowering not yet implemented");
    }
}
