package com.qxotic.jota.runtime;

import java.util.Map;
import java.util.Objects;

public record KernelProgram(
        Kind kind,
        String language,
        Object payload,
        String entryPoint,
        Map<String, String> options) {

    // Language constants
    public static final String JAVA = "java";
    public static final String C = "c";
    public static final String HIP = "hip";
    public static final String CUDA = "cuda";
    public static final String OPENCL = "opencl";

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

    public static KernelProgram source(String language, String source, String entryPoint) {
        return new KernelProgram(Kind.SOURCE, language, source, entryPoint, Map.of());
    }

    public static KernelProgram source(
            String language, String source, String entryPoint, Map<String, String> options) {
        return new KernelProgram(Kind.SOURCE, language, source, entryPoint, options);
    }

    public static KernelProgram binary(String language, byte[] binary, String entryPoint) {
        return new KernelProgram(Kind.BINARY, language, binary, entryPoint, Map.of());
    }

    public static KernelProgram binary(
            String language, byte[] binary, String entryPoint, Map<String, String> options) {
        return new KernelProgram(Kind.BINARY, language, binary, entryPoint, options);
    }
}
