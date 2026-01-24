package ai.qxotic.jota.tensor;

public interface UnaryOp extends Op {

    @Override
    default OpKind kind() {
        return OpKind.ELEMENTWISE;
    }

    UnaryOp IDENTITY = new UnaryOpImpl("identity");

    UnaryOp NEGATE = new UnaryOpImpl("negate");
    UnaryOp ABS = new UnaryOpImpl("abs");
    UnaryOp EXP = new UnaryOpImpl("exp");
    UnaryOp LOG = new UnaryOpImpl("log");
    UnaryOp SQRT = new UnaryOpImpl("sqrt");
    UnaryOp SQUARE = new UnaryOpImpl("square");
    UnaryOp SIN = new UnaryOpImpl("sin");
    UnaryOp COS = new UnaryOpImpl("cos");
    UnaryOp TANH = new UnaryOpImpl("tanh");
    UnaryOp RECIPROCAL = new UnaryOpImpl("reciprocal");

    UnaryOp LOGICAL_NOT = new UnaryOpImpl("logicalNot");
    UnaryOp BITWISE_NOT = new UnaryOpImpl("bitwiseNot");
}

record UnaryOpImpl(String name) implements UnaryOp {}
