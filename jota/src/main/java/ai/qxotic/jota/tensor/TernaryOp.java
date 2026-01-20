package ai.qxotic.jota.tensor;

public interface TernaryOp extends Op {

    @Override
    default OpKind kind() {
        return OpKind.ELEMENTWISE;
    }

    TernaryOp WHERE = new TernaryOpImpl("where");
}

record TernaryOpImpl(String name) implements TernaryOp {}
