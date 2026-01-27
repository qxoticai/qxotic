package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record TransferNode(ExprNode input, Device targetDevice, DataType dataType, Layout layout)
        implements ExprNode {

    @Override
    public Device device() {
        return targetDevice;
    }
}
