package ai.qxotic.jota.tensor;

public interface ReductionOp extends Op {

    @Override
    default OpKind kind() {
        return OpKind.REDUCTION;
    }

    BinaryOp reductionOp();

    ReductionOp SUM = new ReductionOpImpl("sum", BinaryOp.ADD);
    ReductionOp MEAN = new ReductionOpImpl("mean", BinaryOp.ADD);
    ReductionOp MAX = new ReductionOpImpl("max", BinaryOp.MAX);
    ReductionOp MIN = new ReductionOpImpl("min", BinaryOp.MIN);
    ReductionOp PROD = new ReductionOpImpl("prod", BinaryOp.MULTIPLY);
}

record ReductionOpImpl(String name, BinaryOp reductionOp) implements ReductionOp {}
