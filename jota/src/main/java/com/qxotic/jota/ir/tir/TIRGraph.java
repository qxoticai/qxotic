package com.qxotic.jota.ir.tir;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** IR-T graph container. Contains the inputs and outputs for a device-agnostic tensor IR graph. */
public record TIRGraph(List<TIRNode> inputs, List<TIRNode> outputs) {

    public TIRGraph {
        inputs = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(inputs)));
        outputs = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(outputs)));
    }

    public TIRGraph(List<TIRNode> inputs, TIRNode output) {
        this(inputs, List.of(Objects.requireNonNull(output)));
    }

    public TIRGraph(TIRNode input, TIRNode output) {
        this(List.of(Objects.requireNonNull(input)), List.of(Objects.requireNonNull(output)));
    }

    public int inputCount() {
        return inputs.size();
    }

    public int outputCount() {
        return outputs.size();
    }

    @Override
    public String toString() {
        return new TIRTextRenderer().render(this);
    }
}
