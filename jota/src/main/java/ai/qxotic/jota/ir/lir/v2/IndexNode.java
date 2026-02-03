package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public abstract class IndexNode extends V2Node {
    IndexNode(int id, V2Kind kind, V2Node[] inputs, boolean pure, boolean commutative) {
        super(id, kind, DataType.I64, inputs, pure, commutative);
    }
}
