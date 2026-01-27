package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record TernaryNode(
        TernaryOp op,
        ExprNode condition,
        ExprNode trueValue,
        ExprNode falseValue,
        DataType dataType,
        Layout layout,
        Device device)
        implements ExprNode {}
