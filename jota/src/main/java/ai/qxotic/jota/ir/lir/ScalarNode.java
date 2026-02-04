package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public abstract class ScalarNode extends LIRExprNode {
    ScalarNode(
            int id,
            LIRExprKind kind,
            DataType dataType,
            LIRExprNode[] inputs,
            boolean pure,
            boolean commutative) {
        super(id, kind, dataType, inputs, pure, commutative);
    }
}
