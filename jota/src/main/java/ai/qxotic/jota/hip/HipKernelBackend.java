package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRTextRenderer;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.tensor.*;
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

final class HipKernelBackend implements KernelBackend {

    private static final String ENV_HIPCC = "HIPCC";
    private static final Path KERNEL_ROOT = Path.of("__kernels").resolve(Device.HIP.leafName());

    private final KernelExecutableCache cache = new InMemoryKernelCache();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        if (program.language() != KernelProgram.Language.HIP) {
            throw new UnsupportedOperationException("HIP backend expects HIP programs");
        }
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("HIP compile expects source program");
        }
        HipKernelSpec spec = compileSourceProgram(program, cacheKey);
        byte[] hsaco = HipKernelSourceLoader.load(spec.hsacoPath().toString());
        HipModule module = HipModule.load(hsaco);
        HipKernelExecutable exec = new HipKernelExecutable(module.function(spec.kernelName()));
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        if (program.kind() != KernelProgram.Kind.BINARY) {
            throw new UnsupportedOperationException("HIP load expects binary program");
        }
        byte[] hsaco = (byte[]) program.payload();
        HipModule module = HipModule.load(hsaco);
        HipKernelExecutable exec = new HipKernelExecutable(module.function(program.entryPoint()));
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable exec = cache.get(cacheKey);
        if (exec != null) {
            return exec;
        }
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
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 11);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-hip-lir-v1");
            return KernelCacheKey.of(builder.toString());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    @Override
    public KernelExecutableCache cache() {
        return cache;
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

    private HipKernelSpec compileSourceProgram(KernelProgram program, KernelCacheKey key) {
        String kernelName = program.entryPoint();
        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".hip");
        Path hsacoPath = kernelDir.resolve(kernelName + ".hsaco");
        ensureDirectory(kernelDir);
        String source = requireSource(program.payload());
        writeIfChanged(sourcePath, source);
        if (needsCompile(sourcePath, hsacoPath)) {
            compileSource(sourcePath, hsacoPath);
        }
        return new HipKernelSpec(hsacoPath, kernelName);
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected HIP source payload as String");
    }

    private static void ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create HIP kernel directory: " + dir, e);
        }
    }

    private static boolean needsCompile(Path source, Path hsaco) {
        if (!Files.exists(hsaco)) {
            return true;
        }
        try {
            return Files.getLastModifiedTime(source).toMillis()
                    > Files.getLastModifiedTime(hsaco).toMillis();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to stat HIP kernel files", e);
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
            throw new IllegalStateException("Failed to write HIP kernel source: " + path, e);
        }
    }

    private static void compileSource(Path source, Path hsaco) {
        String hipcc = System.getenv(ENV_HIPCC);
        if (hipcc == null || hipcc.isBlank()) {
            hipcc = "hipcc";
        }
        ProcessBuilder builder =
                new ProcessBuilder(
                        hipcc, "--genco", "-O2", source.toString(), "-o", hsaco.toString());
        builder.inheritIO();
        try {
            Process process = builder.start();
            int code = process.waitFor();
            if (code != 0) {
                throw new IllegalStateException(
                        "HIP kernel compilation failed (exit " + code + ")");
            }
        } catch (IOException e) {
            throw new IllegalStateException("HIP kernel compilation failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("HIP kernel compilation interrupted", e);
        }
    }
}
