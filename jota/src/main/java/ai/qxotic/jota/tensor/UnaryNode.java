package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record UnaryNode(UnaryOp op, ExprNode input, DataType dataType, Layout layout, Device device)
        implements ExprNode {}
