package com.qxotic.jota.ir.tir;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** One schedulable kernel unit. */
public record KernelStep(TIRGraph graph, List<ScheduleInputRef> inputs, ValueId output) {

    public KernelStep {
        graph = Objects.requireNonNull(graph, "graph");
        inputs =
                Collections.unmodifiableList(
                        new ArrayList<>(Objects.requireNonNull(inputs, "inputs")));
        output = Objects.requireNonNull(output, "output");
    }
}
