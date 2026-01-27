package ai.qxotic.jota.tensor;

import java.util.Map;
import java.util.Objects;

public record KernelProgram(
        Kind kind, Language language, Object payload, String entryPoint, Map<String, String> options) {

    public KernelProgram {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(language, "language");
        Objects.requireNonNull(payload, "payload");
        Objects.requireNonNull(entryPoint, "entryPoint");
        options = options == null ? Map.of() : Map.copyOf(options);
    }

    public enum Kind {
        SOURCE,
        BINARY
    }

    public enum Language {
        CUDA,
        HIP,
        OPENCL,
        JAVA,
        NATIVE
    }
}
