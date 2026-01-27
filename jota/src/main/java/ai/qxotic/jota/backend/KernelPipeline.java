package ai.qxotic.jota.backend;

import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelHarness;
import ai.qxotic.jota.tensor.KernelProgramGenerator;
import java.util.Objects;

public record KernelPipeline(
        KernelBackend backend,
        KernelProgramGenerator generator,
        KernelHarness harness,
        KernelProgramStore programStore) {

    public KernelPipeline {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(programStore, "programStore");
    }
}
