package ai.qxotic.jota.tensor;

import java.util.List;

public record ExpressionGraph(ExprNode root, List<InputNode> inputs) {

    public ExpressionGraph {
        inputs = List.copyOf(inputs);
    }
}
