package ai.qxotic.jota.tensor;

public interface BinaryOp extends Op {

    @Override
    default OpKind kind() {
        return OpKind.ELEMENTWISE;
    }

    BinaryOp ADD = new BinaryOpImpl("add");
    BinaryOp SUBTRACT = new BinaryOpImpl("subtract");
    BinaryOp MULTIPLY = new BinaryOpImpl("multiply");
    BinaryOp DIVIDE = new BinaryOpImpl("divide");
    BinaryOp MIN = new BinaryOpImpl("min");
    BinaryOp MAX = new BinaryOpImpl("max");
    BinaryOp POW = new BinaryOpImpl("pow");
}

record BinaryOpImpl(String name) implements BinaryOp {}
