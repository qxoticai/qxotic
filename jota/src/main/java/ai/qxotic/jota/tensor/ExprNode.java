package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public sealed interface ExprNode
        permits InputNode, UnaryNode, BinaryNode, ScalarNode, CastNode, ReductionNode {

    DataType dataType();

    Layout layout();

    Device device();
}

record InputNode(int index, DataType dataType, Layout layout, Device device) implements ExprNode {}

record ScalarNode(Number value, DataType dataType, Layout layout, Device device)
        implements ExprNode {}

record UnaryNode(UnaryOp op, ExprNode input, DataType dataType, Layout layout, Device device)
        implements ExprNode {}

record BinaryNode(
        BinaryOp op, ExprNode left, ExprNode right, DataType dataType, Layout layout, Device device)
        implements ExprNode {}

record CastNode(ExprNode input, DataType targetType, Layout layout, Device device)
        implements ExprNode {

    @Override
    public DataType dataType() {
        return targetType;
    }
}

record ReductionNode(
        ReductionOp op,
        ExprNode input,
        int axis,
        boolean keepDims,
        DataType dataType,
        Layout layout,
        Device device)
        implements ExprNode {}
