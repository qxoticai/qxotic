package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record RangeNode(long count, Layout layout, Device device) implements ExprNode {

    @Override
    public DataType dataType() {
        return DataType.I64;
    }
}
