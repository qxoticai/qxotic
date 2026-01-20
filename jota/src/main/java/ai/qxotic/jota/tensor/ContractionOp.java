package ai.qxotic.jota.tensor;

public interface ContractionOp extends Op {

    @Override
    default OpKind kind() {
        return OpKind.CONTRACTION;
    }

    ContractionOp MATMUL = new ContractionOpImpl("matmul");
    ContractionOp BATCHED_MATMUL = new ContractionOpImpl("batched_matmul");
}

record ContractionOpImpl(String name) implements ContractionOp {}
