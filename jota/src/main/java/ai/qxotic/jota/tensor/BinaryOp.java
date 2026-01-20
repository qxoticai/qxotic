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

    BinaryOp LOGICAL_AND = new BinaryOpImpl("logicalAnd");
    BinaryOp LOGICAL_OR = new BinaryOpImpl("logicalOr");
    BinaryOp LOGICAL_XOR = new BinaryOpImpl("logicalXor");

    BinaryOp BITWISE_AND = new BinaryOpImpl("bitwiseAnd");
    BinaryOp BITWISE_OR = new BinaryOpImpl("bitwiseOr");
    BinaryOp BITWISE_XOR = new BinaryOpImpl("bitwiseXor");

    BinaryOp EQUAL = new BinaryOpImpl("equal");
    BinaryOp LESS_THAN = new BinaryOpImpl("lessThan");
}

record BinaryOpImpl(String name) implements BinaryOp {}
