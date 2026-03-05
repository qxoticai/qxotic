package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

final class OpenClKernelBackend implements KernelBackend {

    private static final Path KERNEL_ROOT = Path.of("__kernels").resolve(Device.OPENCL.leafName());
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");

    private final KernelExecutableCache cache = new InMemoryKernelCache();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        if (!KernelProgram.OPENCL.equals(program.language())) {
            throw new UnsupportedOperationException("OpenCL backend expects OpenCL programs");
        }
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("OpenCL compile expects source program");
        }
        OpenClKernelSpec spec = persistSourceProgram(program, cacheKey);
        byte[] sourceBytes = OpenClKernelSourceLoader.load(spec.sourcePath().toString());
        OpenClModule module = OpenClModule.load(sourceBytes);
        return wrapExecutable(
                module, new OpenClKernelExecutable(module.function(spec.kernelName())));
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        if (!KernelProgram.OPENCL.equals(program.language())) {
            throw new UnsupportedOperationException("OpenCL backend expects OpenCL programs");
        }
        if (program.kind() != KernelProgram.Kind.BINARY) {
            throw new UnsupportedOperationException("OpenCL load expects binary program");
        }
        byte[] sourceBytes = (byte[]) program.payload();
        OpenClModule module = OpenClModule.load(sourceBytes);
        return wrapExecutable(
                module, new OpenClKernelExecutable(module.function(program.entryPoint())));
    }

    private static KernelExecutable wrapExecutable(
            OpenClModule module, OpenClKernelExecutable executable) {
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                executable.launch(config, args, stream);
            }

            @Override
            public void close() {
                try {
                    executable.close();
                } finally {
                    module.close();
                }
            }
        };
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable exec = cache.get(cacheKey);
        if (exec != null) {
            log("OpenCL kernel cache hit key=" + cacheKey.value());
            return exec;
        }
        log("OpenCL kernel cache miss key=" + cacheKey.value());
        KernelExecutable created =
                program.kind() == KernelProgram.Kind.BINARY
                        ? load(program, cacheKey)
                        : compile(program, cacheKey);
        cache.put(cacheKey, created);
        return created;
    }

    KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(new LIRTextRenderer().render(graph).getBytes(StandardCharsets.UTF_8));
            digest.update(
                    Long.toString(scratchLayout.totalByteSize()).getBytes(StandardCharsets.UTF_8));
            scratchLayout.offsets().entrySet().stream()
                    .sorted(Comparator.comparingInt(entry -> entry.getKey().id()))
                    .forEach(
                            entry -> {
                                digest.update(
                                        Integer.toString(entry.getKey().id())
                                                .getBytes(StandardCharsets.UTF_8));
                                digest.update(
                                        Long.toString(entry.getValue())
                                                .getBytes(StandardCharsets.UTF_8));
                            });
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 14);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-opencl-lir-v1");
            return KernelCacheKey.of(builder.toString());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    @Override
    public KernelExecutableCache cache() {
        return cache;
    }

    private OpenClKernelSpec persistSourceProgram(KernelProgram program, KernelCacheKey key) {
        String kernelName = program.entryPoint();
        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".cl");
        ensureDirectory(kernelDir);
        String source = requireSource(program.payload());
        writeIfChanged(sourcePath, source);
        return new OpenClKernelSpec(sourcePath, kernelName);
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected OpenCL source payload as String");
    }

    private static void ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create OpenCL kernel directory: " + dir, e);
        }
    }

    private static void writeIfChanged(Path path, String content) {
        try {
            if (Files.exists(path)) {
                String existing = Files.readString(path);
                if (existing.equals(content)) {
                    return;
                }
            }
            Files.writeString(path, content);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write OpenCL kernel source: " + path, e);
        }
    }

    private static final class InMemoryKernelCache implements KernelExecutableCache {
        private final Map<KernelCacheKey, KernelExecutable> map = new ConcurrentHashMap<>();

        @Override
        public KernelExecutable get(KernelCacheKey key) {
            return map.get(key);
        }

        @Override
        public void put(KernelCacheKey key, KernelExecutable exec) {
            map.put(key, exec);
        }
    }

    private static void log(String message) {
        if (KERNEL_LOG) {
            System.out.println("[jota-kernel] " + message);
        }
    }
}
