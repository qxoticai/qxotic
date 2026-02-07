package ai.qxotic.jota.tensor;

import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.ir.tir.TIRVisitor;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

class TIRNodeRemapper implements TIRVisitor<TIRNode> {

    private final int indexOffset;
    private final Map<TIRNode, TIRNode> cache = new IdentityHashMap<>();
    private final List<ai.qxotic.jota.ir.tir.TensorInput> remappedInputs = new ArrayList<>();

    TIRNodeRemapper(int indexOffset) {
        this.indexOffset = indexOffset;
    }

    List<ai.qxotic.jota.ir.tir.TensorInput> remappedInputs() {
        return remappedInputs;
    }

    @Override
    public TIRNode visitTensorInput(ai.qxotic.jota.ir.tir.TensorInput node) {
        return cache.computeIfAbsent(
                node,
                n -> {
                    ai.qxotic.jota.ir.tir.TensorInput remapped =
                            new ai.qxotic.jota.ir.tir.TensorInput(
                                    node.id() + indexOffset, node.dataType(), node.layout());
                    remappedInputs.add(remapped);
                    return remapped;
                });
    }

    @Override
    public TIRNode visitScalarInput(ai.qxotic.jota.ir.tir.ScalarInput node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.ScalarInput(
                                node.id() + indexOffset, node.dataType(), node.shape()));
    }

    @Override
    public TIRNode visitScalarConstant(ai.qxotic.jota.ir.tir.ScalarConstant node) {
        return cache.computeIfAbsent(node, n -> n);
    }

    @Override
    public TIRNode visitIotaConstant(ai.qxotic.jota.ir.tir.IotaConstant node) {
        return cache.computeIfAbsent(node, n -> n);
    }

    @Override
    public TIRNode visitUnaryOp(ai.qxotic.jota.ir.tir.UnaryOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.UnaryOp(
                                node.op(), remap(node.input()), node.shape()));
    }

    @Override
    public TIRNode visitBinaryOp(ai.qxotic.jota.ir.tir.BinaryOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.BinaryOp(
                                node.op(), remap(node.left()), remap(node.right()), node.shape()));
    }

    @Override
    public TIRNode visitTernaryOp(ai.qxotic.jota.ir.tir.TernaryOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.TernaryOp(
                                node.op(),
                                remap(node.cond()),
                                remap(node.trueExpr()),
                                remap(node.falseExpr()),
                                node.shape()));
    }

    @Override
    public TIRNode visitCastOp(ai.qxotic.jota.ir.tir.CastOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.CastOp(
                                remap(node.input()), node.targetDataType(), node.shape()));
    }

    @Override
    public TIRNode visitReductionOp(ai.qxotic.jota.ir.tir.ReductionOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.ReductionOp(
                                node.op(),
                                remap(node.input()),
                                node.axes(),
                                node.keepDims(),
                                node.accumulatorType(),
                                node.shape()));
    }

    @Override
    public TIRNode visitGatherOp(ai.qxotic.jota.ir.tir.GatherOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.GatherOp(
                                remap(node.input()),
                                remap(node.indices()),
                                node.axis(),
                                node.shape()));
    }

    @Override
    public TIRNode visitViewTransform(ai.qxotic.jota.ir.tir.ViewTransform node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.tir.ViewTransform(
                                remap(node.input()),
                                node.kind(),
                                node.layout(),
                                node.needsLazyIndexing()));
    }

    @Override
    public TIRNode visitContiguous(ai.qxotic.jota.ir.tir.Contiguous node) {
        return cache.computeIfAbsent(
                node, n -> new ai.qxotic.jota.ir.tir.Contiguous(remap(node.input()), node.shape()));
    }

    private TIRNode remap(TIRNode node) {
        return node.accept(this);
    }
}
