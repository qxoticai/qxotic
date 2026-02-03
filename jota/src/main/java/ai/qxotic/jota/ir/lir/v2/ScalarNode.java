package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public abstract class ScalarNode extends V2Node {
    ScalarNode(
            int id,
            V2Kind kind,
            DataType dataType,
            V2Node[] inputs,
            boolean pure,
            boolean commutative) {
        super(id, kind, dataType, inputs, pure, commutative);
    }
}
