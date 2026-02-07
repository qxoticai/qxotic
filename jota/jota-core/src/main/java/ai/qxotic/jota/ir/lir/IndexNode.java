package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public abstract class IndexNode extends LIRExprNode {
    IndexNode(int id, LIRExprKind kind, LIRExprNode[] inputs, boolean pure, boolean commutative) {
        super(id, kind, DataType.I64, inputs, pure, commutative);
    }
}
