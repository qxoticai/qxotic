package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.impl.ViewTransforms;

public record ViewTransformOp(
        ExprNode input, ViewTransforms.ViewTransformSpec spec, DataType dataType, Device device)
        implements ExprNode {

    @Override
    public Layout layout() {
        return spec.layout();
    }
}
