package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record BinaryNode(
        BinaryOp op, ExprNode left, ExprNode right, DataType dataType, Layout layout, Device device)
        implements ExprNode {}
