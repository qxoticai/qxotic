package ai.qxotic.jota.ir.tir;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Ordered kernel execution plan derived from a traced TIR graph. */
public record ScheduledProgram(List<KernelStep> steps, ScheduledOutputRef output) {

    public ScheduledProgram {
        steps =
                Collections.unmodifiableList(
                        new ArrayList<>(Objects.requireNonNull(steps, "steps")));
        output = Objects.requireNonNull(output, "output");
    }
}
