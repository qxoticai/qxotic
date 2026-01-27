package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public record ViewTransformOp(
        ExprNode input,
        Layout layout,
        long byteOffsetDelta,
        String hint,
        DataType dataType,
        Device device)
        implements ExprNode {}
