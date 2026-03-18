package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

final class MetalKernelBackend implements KernelBackend {

    private static final Path KERNEL_ROOT = KernelCachePaths.deviceRoot(DeviceType.METAL);
    private static final String COMPILER_PROPERTY = "jota.metal.compiler";
    private static final String COMPILER_ENV = "JOTA_METAL_COMPILER";
    private static final String METAL_COMPILER = resolveCompilerExecutable();
    private static final String METAL_COMPILE_FLAGS_PROPERTY = "jota.metal.compile.flags";
    private static final String LINK_FLAGS_PROPERTY = "jota.metal.link.flags";
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.metal.compile.timeout.seconds", 15L);
    private static final String OPT_LEVEL =
            System.getProperty("jota.metal.compile.opt", "2").trim();
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");

    private final KernelExecutableCache cache = new InMemoryKernelCache();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        if (!"metal".equals(program.language())) {
            throw new UnsupportedOperationException("Metal backend expects Metal programs");
        }
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("Metal compile expects source program");
        }
        MetalKernelSpec spec = compileSourceProgram(program, cacheKey);
        byte[] metallib = MetalKernelSourceLoader.load(spec.metallibPath().toString());
        MetalModule module = MetalModule.load(metallib);
        return wrapExecutable(
                module, new MetalKernelExecutable(module.function(spec.kernelName())));
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        if (!"metal".equals(program.language())) {
            throw new UnsupportedOperationException("Metal backend expects Metal programs");
        }
        if (program.kind() != KernelProgram.Kind.BINARY) {
            throw new UnsupportedOperationException("Metal load expects binary program");
        }
        byte[] metallib = (byte[]) program.payload();
        MetalModule module = MetalModule.load(metallib);
        MetalKernelExecutable executable =
                new MetalKernelExecutable(module.function(program.entryPoint()));
        return wrapExecutable(module, executable);
    }

    private static KernelExecutable wrapExecutable(
            MetalModule module, MetalKernelExecutable executable) {
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
            log("Metal kernel cache hit key=" + cacheKey.value());
            return exec;
        }
        log("Metal kernel cache miss key=" + cacheKey.value());
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
            digest.update(METAL_COMPILER.getBytes(StandardCharsets.UTF_8));
            digest.update(OPT_LEVEL.getBytes(StandardCharsets.UTF_8));
            digest.update(
                    String.join(" ", extraFlags(METAL_COMPILE_FLAGS_PROPERTY))
                            .getBytes(StandardCharsets.UTF_8));
            digest.update(
                    String.join(" ", extraFlags(LINK_FLAGS_PROPERTY))
                            .getBytes(StandardCharsets.UTF_8));
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 13);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-metal-lir-v1");
            return KernelCacheKey.of(builder.toString());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    @Override
    public KernelExecutableCache cache() {
        return cache;
    }

    private MetalKernelSpec compileSourceProgram(KernelProgram program, KernelCacheKey key) {
        String kernelName = program.entryPoint();
        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".metal");
        Path airPath = kernelDir.resolve(kernelName + ".air");
        Path metallibPath = kernelDir.resolve(kernelName + ".metallib");
        ensureDirectory(kernelDir);
        String source = requireSource(program.payload());
        writeIfChanged(sourcePath, source);
        if (needsCompile(sourcePath, metallibPath)) {
            log("Metal kernel compile key=" + key.value() + " entry=" + kernelName);
            compileSource(sourcePath, airPath, metallibPath);
        } else {
            log("Metal kernel reuse key=" + key.value() + " entry=" + kernelName);
        }
        return new MetalKernelSpec(metallibPath, kernelName);
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected Metal source payload as String");
    }

    private static void compileSource(Path source, Path air, Path metallib) {
        List<String> metalCompile = new ArrayList<>();
        metalCompile.add(METAL_COMPILER);
        metalCompile.add("-sdk");
        metalCompile.add("macosx");
        metalCompile.add("metal");
        metalCompile.add("-O" + OPT_LEVEL);
        metalCompile.addAll(extraFlags(METAL_COMPILE_FLAGS_PROPERTY));
        metalCompile.add("-c");
        metalCompile.add(source.toString());
        metalCompile.add("-o");
        metalCompile.add(air.toString());
        runCommand(metalCompile, "Metal source compilation failed");

        List<String> libCompile = new ArrayList<>();
        libCompile.add(METAL_COMPILER);
        libCompile.add("-sdk");
        libCompile.add("macosx");
        libCompile.add("metallib");
        libCompile.addAll(extraFlags(LINK_FLAGS_PROPERTY));
        libCompile.add(air.toString());
        libCompile.add("-o");
        libCompile.add(metallib.toString());
        runCommand(libCompile, "Metal library packaging failed");
    }

    private static void runCommand(List<String> command, String errorPrefix) {
        ProcessBuilder builder = new ProcessBuilder(command);
        try {
            Process process = builder.start();
            StreamCapture stdout = StreamCapture.start(process.getInputStream());
            StreamCapture stderr = StreamCapture.start(process.getErrorStream());
            if (!process.waitFor(COMPILE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                String out = stdout.await();
                String err = stderr.await();
                throw new IllegalStateException(
                        errorPrefix
                                + ": timed out after "
                                + COMPILE_TIMEOUT_SECONDS
                                + "s\n"
                                + formatCommandFailure(command, -1, out, err));
            }
            int code = process.exitValue();
            String out = stdout.await();
            String err = stderr.await();
            if (code != 0) {
                throw new IllegalStateException(
                        errorPrefix + "\n" + formatCommandFailure(command, code, out, err));
            }
        } catch (IOException e) {
            throw new IllegalStateException(errorPrefix, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(errorPrefix + " (interrupted)", e);
        }
    }

    private static String formatCommandFailure(
            List<String> command, int exitCode, String stdout, String stderr) {
        StringBuilder details = new StringBuilder();
        details.append("command: ").append(joinCommand(command)).append('\n');
        if (exitCode >= 0) {
            details.append("exit: ").append(exitCode).append('\n');
        }
        if (!stdout.isBlank()) {
            details.append("stdout:\n").append(limitOutput(stdout)).append('\n');
        }
        if (!stderr.isBlank()) {
            details.append("stderr:\n").append(limitOutput(stderr)).append('\n');
        }
        return details.toString().trim();
    }

    private static String joinCommand(List<String> command) {
        return String.join(" ", command);
    }

    private static String limitOutput(String text) {
        int max = 4000;
        if (text.length() <= max) {
            return text;
        }
        return text.substring(0, max) + "\n... [truncated " + (text.length() - max) + " chars]";
    }

    private static List<String> extraFlags(String propertyName) {
        String value = System.getProperty(propertyName);
        if (value == null || value.isBlank()) {
            return List.of();
        }
        String[] parts = value.trim().split("\\s+");
        List<String> result = new ArrayList<>(parts.length);
        for (String part : parts) {
            if (!part.isBlank()) {
                result.add(part);
            }
        }
        return List.copyOf(result);
    }

    private static String resolveCompilerExecutable() {
        String fromProperty = System.getProperty(COMPILER_PROPERTY);
        if (fromProperty != null && !fromProperty.isBlank()) {
            return fromProperty.trim();
        }
        String fromEnv = System.getenv(COMPILER_ENV);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return fromEnv.trim();
        }
        return "xcrun";
    }

    private static final class StreamCapture {
        private Thread thread;
        private final ByteArrayOutputStream output = new ByteArrayOutputStream();
        private volatile IOException error;

        private StreamCapture() {}

        static StreamCapture start(InputStream input) {
            StreamCapture capture = new StreamCapture();
            capture.thread =
                    new Thread(
                            () -> {
                                try (input) {
                                    input.transferTo(capture.output);
                                } catch (IOException e) {
                                    capture.error = e;
                                }
                            },
                            "jota-metal-cmd-capture");
            capture.thread.setDaemon(true);
            capture.thread.start();
            return capture;
        }

        String await() throws InterruptedException, IOException {
            thread.join();
            if (error != null) {
                throw error;
            }
            return output.toString(StandardCharsets.UTF_8);
        }
    }

    private static void ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create Metal kernel directory: " + dir, e);
        }
    }

    private static boolean needsCompile(Path source, Path metallib) {
        if (!Files.exists(metallib)) {
            return true;
        }
        try {
            return Files.getLastModifiedTime(source).toMillis()
                    > Files.getLastModifiedTime(metallib).toMillis();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to stat Metal kernel files", e);
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
            throw new IllegalStateException("Failed to write Metal kernel source: " + path, e);
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
