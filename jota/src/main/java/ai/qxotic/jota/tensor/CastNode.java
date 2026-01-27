package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record CastNode(ExprNode input, DataType targetType, Layout layout, Device device)
        implements ExprNode {

    @Override
    public DataType dataType() {
        return targetType;
    }
}
