package ai.qxotic.jota.tensor;

import ai.qxotic.jota.ir.irt.IRTNode;
import ai.qxotic.jota.ir.irt.IRTVisitor;
import java.util.IdentityHashMap;
import java.util.Map;

class IRTNodeRemapper implements IRTVisitor<IRTNode> {

    private final int indexOffset;
    private final Map<IRTNode, IRTNode> cache = new IdentityHashMap<>();

    IRTNodeRemapper(int indexOffset) {
        this.indexOffset = indexOffset;
    }

    @Override
    public IRTNode visitTensorInput(ai.qxotic.jota.ir.irt.TensorInput node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.irt.TensorInput(
                                node.id() + indexOffset, node.dataType(), node.layout()));
    }

    @Override
    public IRTNode visitScalarConstant(ai.qxotic.jota.ir.irt.ScalarConstant node) {
        return cache.computeIfAbsent(node, n -> n);
    }

    @Override
    public IRTNode visitIotaConstant(ai.qxotic.jota.ir.irt.IotaConstant node) {
        return cache.computeIfAbsent(node, n -> n);
    }

    @Override
    public IRTNode visitUnaryOp(ai.qxotic.jota.ir.irt.UnaryOp node) {
        return cache.computeIfAbsent(
                node, n -> new ai.qxotic.jota.ir.irt.UnaryOp(node.op(), remap(node.input())));
    }

    @Override
    public IRTNode visitBinaryOp(ai.qxotic.jota.ir.irt.BinaryOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.irt.BinaryOp(
                                node.op(), remap(node.left()), remap(node.right())));
    }

    @Override
    public IRTNode visitTernaryOp(ai.qxotic.jota.ir.irt.TernaryOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.irt.TernaryOp(
                                node.op(),
                                remap(node.cond()),
                                remap(node.trueExpr()),
                                remap(node.falseExpr())));
    }

    @Override
    public IRTNode visitCastOp(ai.qxotic.jota.ir.irt.CastOp node) {
        return cache.computeIfAbsent(
                node,
                n -> new ai.qxotic.jota.ir.irt.CastOp(remap(node.input()), node.targetDtype()));
    }

    @Override
    public IRTNode visitReductionOp(ai.qxotic.jota.ir.irt.ReductionOp node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.irt.ReductionOp(
                                node.op(), remap(node.input()), node.axes(), node.keepDims()));
    }

    @Override
    public IRTNode visitViewTransform(ai.qxotic.jota.ir.irt.ViewTransform node) {
        return cache.computeIfAbsent(
                node,
                n ->
                        new ai.qxotic.jota.ir.irt.ViewTransform(
                                remap(node.input()), node.hint(), node.layout()));
    }

    @Override
    public IRTNode visitContiguous(ai.qxotic.jota.ir.irt.Contiguous node) {
        return cache.computeIfAbsent(
                node, n -> new ai.qxotic.jota.ir.irt.Contiguous(remap(node.input())));
    }

    private IRTNode remap(IRTNode node) {
        return node.accept(this);
    }
}
