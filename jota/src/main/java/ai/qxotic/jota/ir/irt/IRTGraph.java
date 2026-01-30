package ai.qxotic.jota.ir.irt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** IR-T graph container. Contains the inputs and outputs for a device-agnostic tensor IR graph. */
public record IRTGraph(List<IRTNode> inputs, List<IRTNode> outputs) {

    public IRTGraph {
        inputs = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(inputs)));
        outputs = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(outputs)));
    }

    public IRTGraph(List<IRTNode> inputs, IRTNode output) {
        this(inputs, List.of(Objects.requireNonNull(output)));
    }

    public IRTGraph(IRTNode input, IRTNode output) {
        this(List.of(Objects.requireNonNull(input)), List.of(Objects.requireNonNull(output)));
    }

    public int inputCount() {
        return inputs.size();
    }

    public int outputCount() {
        return outputs.size();
    }
}
