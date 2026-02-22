package com.qxotic.jota.runtime;

import com.qxotic.jota.tensor.KernelBackend;
import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelExecutable;
import com.qxotic.jota.tensor.KernelProgram;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public record KernelService(
        KernelBackend backend,
        KernelProgramStore sourceStore,
        KernelProgramStore binaryStore,
        ConcurrentMap<String, KernelCacheKey> namedKernels) {

    public KernelService(
            KernelBackend backend, KernelProgramStore sourceStore, KernelProgramStore binaryStore) {
        this(backend, sourceStore, binaryStore, new ConcurrentHashMap<>());
    }

    public KernelService {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(sourceStore, "sourceStore");
        Objects.requireNonNull(binaryStore, "binaryStore");
        namedKernels = namedKernels == null ? new ConcurrentHashMap<>() : namedKernels;
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

    public KernelExecutable register(String name, KernelProgram program, KernelCacheKey key) {
        Objects.requireNonNull(name, "name");
        KernelExecutable executable = register(program, key);
        namedKernels.put(name, key);
        return executable;
    }

    public Optional<KernelProgram> loadRegisteredKernel(String name) {
        Objects.requireNonNull(name, "name");
        KernelCacheKey key = namedKernels.get(name);
        if (key == null) {
            return Optional.empty();
        }
        return loadRegisteredKernel(key);
    }

    public Optional<KernelExecutable> loadRegisteredExecutable(String name) {
        Objects.requireNonNull(name, "name");
        KernelCacheKey key = namedKernels.get(name);
        if (key == null) {
            return Optional.empty();
        }
        Optional<KernelProgram> program = loadRegisteredKernel(key);
        return program.map(p -> backend.getOrCompile(p, key));
    }

    public Optional<KernelExecutable> loadRegisteredBinaryExecutable(String name) {
        Objects.requireNonNull(name, "name");
        KernelCacheKey key = namedKernels.get(name);
        if (key == null) {
            return Optional.empty();
        }
        Optional<KernelProgram> program = loadRegisteredBinary(key);
        return program.map(p -> backend.getOrCompile(p, key));
    }

    public void bindKernelName(String name, KernelCacheKey key) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(key, "key");
        namedKernels.put(name, key);
    }

    public Map<String, KernelCacheKey> namedKernelKeys() {
        return Map.copyOf(namedKernels);
    }
}
