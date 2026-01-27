package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;

public sealed interface ExprNode
        permits InputNode,
                UnaryNode,
                BinaryNode,
                TernaryNode,
                TransferNode,
                ContiguousNode,
                ScalarNode,
                RangeNode,
                CastNode,
                ViewTransformOp,
                ReductionNode {

    DataType dataType();

    Layout layout();

    Device device();
}
