package ai.qxotic.jota.backend;

import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import java.util.Objects;
import java.util.Optional;

public record KernelService(
        KernelBackend backend, KernelProgramStore sourceStore, KernelProgramStore binaryStore) {

    public KernelService {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(sourceStore, "sourceStore");
        Objects.requireNonNull(binaryStore, "binaryStore");
    }

    public KernelExecutable register(KernelProgram program, KernelCacheKey key) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(key, "key");
        if (program.kind() == KernelProgram.Kind.BINARY) {
            binaryStore.store(program, key);
        } else {
            sourceStore.store(program, key);
        }
        return backend.getOrCompile(program, key);
    }

    public Optional<KernelProgram> loadRegisteredKernel(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        Optional<KernelProgram> binary = binaryStore.load(key);
        if (binary.isPresent()) {
            return binary;
        }
        return sourceStore.load(key);
    }

    public Optional<KernelProgram> loadRegisteredSource(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        return sourceStore.load(key);
    }

    public Optional<KernelProgram> loadRegisteredBinary(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        return binaryStore.load(key);
    }
}
